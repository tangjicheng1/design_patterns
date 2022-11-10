#include "utils.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

const int run_loop = 100;

struct DevBuff final {
  void* data;
  int size;
};

int dims2size(nvinfer1::Dims dims) {
  int size = 1;
  for (int i = 0; i < dims.nbDims; i++) {
    if (dims.d[i] <= 0) {
      printf("Error: TensorRT Dims should NOT <= 0\n");
      abort();
    }
    size *= dims.d[i];
  }
  return size;
}

class NoLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
    // do NOT need output log
    printf("[TensorRT] %s\n", msg);
    return;
  }
};

NoLogger g_logger;

// std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;

void test_infer(const char* model_filename) {
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(g_logger));
  unsigned int explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  config->setProfileStream(0);
  nvinfer1::IOptimizationProfile* prof(builder->createOptimizationProfile());
  config->addOptimizationProfile(prof);

  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, g_logger));
  CHECK_POINTER(parser);
  bool is_parsed = parser->parseFromFile(model_filename, 1);
  if (!is_parsed) {
    printf("Error: parse onnx failed.\n");
    abort();
  }

  std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(g_logger));
  std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(), plan->size()));
  std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());

  int input_count = network->getNbInputs();
  int output_count = network->getNbOutputs();
  std::vector<DevBuff> input_buff(input_count);
  std::vector<DevBuff> output_buff(output_count);
  for (int i = 0; i < input_count; i++) {
    nvinfer1::Dims cur_input_dims = network->getInput(i)->getDimensions();
    // context->setBindingDimensions(i, cur_input_dims);
    input_buff[i].size = dims2size(cur_input_dims);
    void* buff = nullptr;
    cudaMalloc(&buff, input_buff[i].size * sizeof(float));
    std::vector<float> cpu_data = generate_data(input_buff[i].size, 0);
    cudaMemcpy(buff, cpu_data.data(), input_buff[i].size * sizeof(float), cudaMemcpyHostToDevice);
    input_buff[i].data = buff;
  }
  for (int i = 0; i < output_count; i++) {
    nvinfer1::Dims cur_output_dims = network->getOutput(i)->getDimensions();
    // context->setBindingDimensions(i + input_count, cur_output_dims);
    output_buff[i].size = dims2size(cur_output_dims);
    void* buff = nullptr;
    cudaMalloc(&buff, output_buff[i].size * sizeof(float));
    output_buff[i].data = buff;
  }

  std::vector<void*> all_buff;
  all_buff.reserve(input_count + output_count);
  for (const auto& iter : input_buff) {
    all_buff.push_back(iter.data);
  }
  for (const auto& iter : output_buff) {
    all_buff.push_back(iter.data);
  }

  bool is_ok;
  int count = 0;
  double sum_cost = 0.0;
  for (int i = 0; i < run_loop; i++) {
    cudaDeviceSynchronize();
    double start_ms = now_ms();
    is_ok = context->executeV2(all_buff.data());
    cudaDeviceSynchronize();
    double end_ms = now_ms();

    double cur_cost = end_ms - start_ms;
    if (i > 10) {
      count += 1;
      sum_cost += cur_cost;
    }

    if (!is_ok) {
      printf("Error: TensorRT runtime error\n");
      abort();
    } else {
      printf("cost: %lf\n", cur_cost);
    }
  }

  if (count > 0) {
    printf("average cost: %lf\n", sum_cost / count);
  }

  for (auto iter : all_buff) {
    cudaFree(iter);
  }

  return;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("usage: %s model.onnx\n", argv[0]);
    return 1;
  }

  test_infer(argv[1]);

  return 0;
}