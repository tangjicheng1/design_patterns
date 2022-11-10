#pragma once

#include "onnxruntime/core/session/onnxruntime_c_api.h"

#include "cuda_runtime_api.h"

#include <stdio.h>
#include <time.h>
#include <iostream>
#include <string>
#include <vector>

#define ORT_ABORT_ON_ERROR(expr)                                 \
  do {                                                           \
    OrtStatus* this_ort_status = (expr);                         \
    if (this_ort_status != NULL) {                               \
      const char* msg = g_ort->GetErrorMessage(this_ort_status); \
      fprintf(stderr, "%s\n", msg);                              \
      g_ort->ReleaseStatus(this_ort_status);                     \
      abort();                                                   \
    }                                                            \
  } while (0)

#define CHECK_RET(value)                                       \
  do {                                                         \
    if (value != 0) {                                          \
      fprintf(stderr, "[Error] %s: %d\n", __FILE__, __LINE__); \
      abort();                                                 \
    }                                                          \
  } while (0)

#define CHECK_POINTER(p)                                       \
  do {                                                         \
    if ((p) == nullptr) {                                      \
      fprintf(stderr, "[Error] %s: %d\n", __FILE__, __LINE__); \
      abort();                                                 \
    }                                                          \
  } while (0)

// 防止与onnxruntime的TensorInfo重名
struct MyTensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  void* data;
  // only support data_type == fp32
};

inline std::vector<float> generate_data(size_t size, int mode) {
  std::vector<float> result(size);
  for (size_t i = 0; i < size; i++) {
    result[i] = 1.0f;
  }
  return result;
}

inline void fix_shape(std::vector<int64_t>& shape) {
  for (auto& iter : shape) {
    if (iter <= 0) {
      iter = 1;
    }
  }
  return;
}

inline int64_t shape2size(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (auto iter : shape) {
    if (iter <= 0) {
      printf("Do NOT support shape[i] <= 0\n");
      abort();
    }
    size *= iter;
  }
  return size;
}

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

inline void print_tensor_info(const std::vector<MyTensorInfo>& tensor_info) {
  for (size_t i = 0; i < tensor_info.size(); i++) {
    std::cout << "index: " << i << " name: " << tensor_info[i].name << " shape: ";
    for (auto iter : tensor_info[i].shape) {
      std::cout << iter << " ";
    }
    std::cout << std::endl;
  }
  return;
}

inline double now_ms() {
  struct timespec res;
  clock_gettime(CLOCK_REALTIME, &res);
  return 1000.0 * res.tv_sec + (double)res.tv_nsec / 1e6;
}