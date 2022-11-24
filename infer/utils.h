#pragma once
#include "cpu_utils.h"
#include "cuda_runtime_api.h"

inline void cuda_alloc(std::vector<MyTensorInfo>& tensor_info) {
  for (auto& iter : tensor_info) {
    int64_t size = shape2size(iter.shape);
    void* gpu_data = nullptr;
    cudaMalloc(&gpu_data, size * sizeof(float));
    std::vector<float> cpu_data = generate_data(size, 0);
    cudaMemcpy(gpu_data, cpu_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    iter.data = gpu_data;
  }
  return;
}

inline void cuda_free(std::vector<MyTensorInfo>& tensor_info) {
  for (auto& iter : tensor_info) {
    cudaFree(iter.data);
  }
  return;
}
