#include "onnxruntime/core/session/onnxruntime_c_api.h"

#include "utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

const OrtApi* g_ort = nullptr;

// onnxruntime default cpu memory allocator, use posix_memalign/free in linux
// Always returns the same pointer to the same default allocator.
// should NOT be freed
OrtAllocator* default_allocator = nullptr;

const int run_loop = 100;

void enable_cuda(OrtSessionOptions* session_options) {
  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = 0;
  // memset(&cuda_options, 0, sizeof(cuda_options));  // this is for pure C language, set each option is zero.
  cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  // cuda_options.gpu_mem_limit = SIZE_MAX; // this is default value.
  ORT_ABORT_ON_ERROR(g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &cuda_options));
  return;
}

void set_input_or_output_tensor_info(OrtSession* session, std::vector<MyTensorInfo>& tensor_info, bool is_input) {
  tensor_info.clear();
  size_t count = 0;
  if (is_input) {
    ORT_ABORT_ON_ERROR(g_ort->SessionGetInputCount(session, &count));
  } else {
    ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputCount(session, &count));
  }
  tensor_info.resize(count);
  for (size_t i = 0; i < count; i++) {
    // set tensor name
    char* tensor_name = nullptr;
    if (is_input) {
      ORT_ABORT_ON_ERROR(g_ort->SessionGetInputName(session, i, default_allocator, &tensor_name));
    } else {
      ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputName(session, i, default_allocator, &tensor_name));
    }
    tensor_info[i].name = std::string(tensor_name);
    default_allocator->Free(default_allocator, tensor_name);

    // set input tensor data_type and shape
    OrtTypeInfo* type_info = nullptr;
    const OrtTensorTypeAndShapeInfo* cur_tensor_info = nullptr;
    if (is_input) {
      ORT_ABORT_ON_ERROR(g_ort->SessionGetInputTypeInfo(session, i, &type_info));
    } else {
      ORT_ABORT_ON_ERROR(g_ort->SessionGetOutputTypeInfo(session, i, &type_info));
    }
    ORT_ABORT_ON_ERROR(g_ort->CastTypeInfoToTensorInfo(type_info, &cur_tensor_info));
    ONNXTensorElementDataType data_type;
    ORT_ABORT_ON_ERROR(g_ort->GetTensorElementType(cur_tensor_info, &data_type));
    if (data_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      printf("Error: only support input or output data_type == fp32\n");
      abort();
    }
    size_t dims_len = 0;
    ORT_ABORT_ON_ERROR(g_ort->GetDimensionsCount(cur_tensor_info, &dims_len));
    tensor_info[i].shape.resize(dims_len);
    ORT_ABORT_ON_ERROR(g_ort->GetDimensions(cur_tensor_info, tensor_info[i].shape.data(), dims_len));
    fix_shape(tensor_info[i].shape);
    g_ort->ReleaseTypeInfo(type_info);
  }

  return;
}

int run_inference(OrtSession* session) {
  int ret = 0;
  // get inputs & outputs informations
  std::vector<MyTensorInfo> input_tensor;
  std::vector<MyTensorInfo> output_tensor;
  set_input_or_output_tensor_info(session, input_tensor, true);
  set_input_or_output_tensor_info(session, output_tensor, false);

  std::cout << "**** input ****\n";
  print_tensor_info(input_tensor);
  std::cout << "**** output ****\n";
  print_tensor_info(output_tensor);

  cuda_alloc(input_tensor);
  cuda_alloc(output_tensor);

#ifdef USE_CPU_DATA
// TODO
#else
  OrtIoBinding* io_binding = nullptr;
  ORT_ABORT_ON_ERROR(g_ort->CreateIoBinding(session, &io_binding));

  // IoBinding to cuda memory
  OrtMemoryInfo* cuda_memory_info;
  int cuda_memory_info_id = 0;
  ORT_ABORT_ON_ERROR(
      g_ort->CreateMemoryInfo("Cuda", OrtDeviceAllocator, cuda_memory_info_id, OrtMemTypeDefault, &cuda_memory_info));
  std::vector<OrtValue*> input_ortvalue(input_tensor.size(), nullptr);
  std::vector<OrtValue*> output_ortvalue(output_tensor.size(), nullptr);
  for (size_t i = 0; i < input_tensor.size(); i++) {
    int64_t cur_tensor_size = shape2size(input_tensor[i].shape) * sizeof(float);
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(cuda_memory_info, input_tensor[i].data, cur_tensor_size,
                                                             input_tensor[i].shape.data(), input_tensor[i].shape.size(),
                                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_ortvalue[i]));
    ORT_ABORT_ON_ERROR(g_ort->BindInput(io_binding, input_tensor[i].name.c_str(), input_ortvalue[i]));
  }
  for (size_t i = 0; i < output_tensor.size(); i++) {
    int64_t cur_tensor_size = shape2size(output_tensor[i].shape) * sizeof(float);
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
        cuda_memory_info, output_tensor[i].data, cur_tensor_size, output_tensor[i].shape.data(),
        output_tensor[i].shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &output_ortvalue[i]));
    ORT_ABORT_ON_ERROR(g_ort->BindOutput(io_binding, output_tensor[i].name.c_str(), output_ortvalue[i]));
  }

  OrtRunOptions* run_options = nullptr;

  int run_count = 0;
  double sum_cost = 0.0;
  for (int i = 0; i < run_loop; i++) {
    cudaDeviceSynchronize();
    double start_ms = now_ms();
    ORT_ABORT_ON_ERROR(g_ort->RunWithBinding(session, run_options, io_binding));
    cudaDeviceSynchronize();
    double end_ms = now_ms();
    double cost_ms = end_ms - start_ms;
    printf("cost: %lf\n", cost_ms);
    if (i > 10) {
      run_count += 1;
      sum_cost += cost_ms;
    }
  }

  if (run_count > 0) {
    printf("average cost: %lf\n", sum_cost / run_count);
  }

  // free ortvalue
  for (auto& iter : input_ortvalue) {
    g_ort->ReleaseValue(iter);
  }
  for (auto& iter : output_ortvalue) {
    g_ort->ReleaseValue(iter);
  }
  g_ort->ReleaseMemoryInfo(cuda_memory_info);
  g_ort->ReleaseIoBinding(io_binding);
#endif

  cuda_free(input_tensor);
  cuda_free(output_tensor);

  return ret;
}

void test_infer(const char* model_filename) {
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test_infer", &env));
  OrtSessionOptions* session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
  assert(env != nullptr && session_options != nullptr);
  enable_cuda(session_options);

  OrtSession* session;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_filename, session_options, &session));

  run_inference(session);

  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);
  return;
}

int main(int argc, char** argv) {
  std::cout << "ORT_API_VERSION: " << ORT_API_VERSION << std::endl;
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (g_ort == nullptr) {
    printf("Failed to init onnxruntime engine!\n");
    return 1;
  }
  ORT_ABORT_ON_ERROR(g_ort->GetAllocatorWithDefaultOptions(&default_allocator));

  if (argc != 2) {
    printf("usage: %s model.onnx\n", argv[0]);
    return 1;
  }

  test_infer(argv[1]);

  return 0;
}