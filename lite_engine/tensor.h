#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <vector>

inline int shape2size(const std::vector<int>& shape) {
  int ret = 1;
  for (auto iter : shape) {
    ret *= iter;
  }
  return ret;
}

class Tensor final {
 public:
  Tensor() : data_(nullptr) {}
  Tensor(const std::vector<int>& shape) {
    int size = shape2size(shape);
    cudaMalloc((void**)&data_, sizeof(float) * size);
    shape_ = shape;
  }
  ~Tensor() { free(); }

  void resize(const std::vector<int>& shape) {
    int new_size = shape2size(shape);
    free();
    cudaMalloc((void**)&data_, new_size);
    shape_ = shape;
  }

  std::vector<int> shape() { return shape_; }
  void* data() { return data_; }

  const void* data() const { return data_; }

  void free() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  void copy_from_host(const std::vector<float>& cpu_data) {
    cudaMemcpy(data_, cpu_data.data(), shape2size(shape_) * sizeof(float), cudaMemcpyHostToDevice);
  }

 private:
  std::vector<int> shape_;
  float* data_ = nullptr;
};