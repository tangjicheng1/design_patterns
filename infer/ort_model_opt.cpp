#include "onnxruntime/core/session/onnxruntime_c_api.h"

#include "cpu_utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

const OrtApi* g_ort = nullptr;
OrtAllocator* default_allocator = nullptr;

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

std::vector<const char*> get_tensor_names(const std::vector<MyTensorInfo>& input_tensor) {
  std::vector<const char*> ret(input_tensor.size(), nullptr);
  for (size_t i = 0; i < input_tensor.size(); i++) {
    ret[i] = input_tensor[i].name.c_str();
  }
  return ret;
}

int run_inference(OrtSession* session) {
  int ret = 0;
  // get inputs & outputs informations
  OrtRunOptions* run_options = nullptr;

  // get inputs & outputs informations
  std::vector<MyTensorInfo> input_tensor;
  std::vector<MyTensorInfo> output_tensor;
  set_input_or_output_tensor_info(session, input_tensor, true);
  set_input_or_output_tensor_info(session, output_tensor, false);

  std::cout << "**** input ****\n";
  print_tensor_info(input_tensor);
  std::cout << "**** output ****\n";
  print_tensor_info(output_tensor);

  std::vector<const char*> input_names = get_tensor_names(input_tensor);
  std::vector<const char*> output_names = get_tensor_names(output_tensor);

  cpu_alloc(input_tensor);
  cpu_alloc(output_tensor);

  std::vector<OrtValue*> input_ortvalue(input_tensor.size(), nullptr);
  std::vector<OrtValue*> output_ortvalue(output_tensor.size(), nullptr);

  OrtMemoryInfo* cpu_memory_info;
  int cpu_memory_info_id = 0;
  ORT_ABORT_ON_ERROR(
      g_ort->CreateMemoryInfo("Cpu", OrtArenaAllocator, cpu_memory_info_id, OrtMemTypeDefault, &cpu_memory_info));

  for (size_t i = 0; i < input_tensor.size(); i++) {
    int64_t cur_tensor_size = shape2size(input_tensor[i].shape) * sizeof(float);
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(cpu_memory_info, input_tensor[i].data, cur_tensor_size,
                                                             input_tensor[i].shape.data(), input_tensor[i].shape.size(),
                                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_ortvalue[i]));
  }
  for (size_t i = 0; i < output_tensor.size(); i++) {
    int64_t cur_tensor_size = shape2size(output_tensor[i].shape) * sizeof(float);
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(
        cpu_memory_info, output_tensor[i].data, cur_tensor_size, output_tensor[i].shape.data(),
        output_tensor[i].shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &output_ortvalue[i]));
  }

  double start_ms = now_ms();
  ORT_ABORT_ON_ERROR(g_ort->Run(session, run_options, input_names.data(), input_ortvalue.data(), input_tensor.size(),
                                output_names.data(), output_tensor.size(), output_ortvalue.data()));
  double end_ms = now_ms();
  double cost_ms = end_ms - start_ms;

  printf("cpu first run cost: %lf\n", cost_ms);

  cpu_free(input_tensor);
  cpu_free(output_tensor);

  return ret;
}

void test_infer(const char* model_filename, const char* opt_model_filename) {
  OrtEnv* env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test_infer", &env));
  OrtSessionOptions* session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));
  assert(env != nullptr && session_options != nullptr);
  ORT_ABORT_ON_ERROR(g_ort->SetOptimizedModelFilePath(session_options, opt_model_filename));

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

  if (argc != 3) {
    printf("usage: %s model.onnx opt_model.onnx\n", argv[0]);
    return 1;
  }

  test_infer(argv[1], argv[2]);

  return 0;
}