#pragma once

#include <cudnn.h>
#include <stddef.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDNN(expression)                                                       \
  do {                                                                                \
    cudnnStatus_t status = (expression);                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                                             \
      printf("[%s:%d] Error: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

// GPU
struct Tensor {
  void* data;
  std::vector<size_t> shape;
};

// CPU
struct Matrix {
  std::vector<float> data;
  std::vector<size_t> shape;
};

inline size_t shape2size(const std::vector<size_t>& shape) {
  size_t ret = 1;
  for (auto iter : shape) {
    ret *= iter;
  }
  return ret;
}

inline std::vector<float> read_data(size_t size, const std::string& filename) {
  std::vector<float> data(size);
  std::ifstream ifs(filename);
  for (int i = 0; i < size; i++) {
    ifs >> data[i];
  }
  return data;
}

inline void write_data(const std::vector<float>& data, size_t size, const std::string& filename) {
  std::ofstream ofs(filename);
  for (int i = 0; i < size; i++) {
    if (i != 0 && i % 16 == 0) {
      ofs << "\n";
    }
    ofs << std::setprecision(6) << data[i] << " ";
  }
}

inline void print(const Matrix& mat) {
  std::cout << "shape: ";
  for (auto iter : mat.shape) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
  std::cout << "data: ";
  for (auto iter : mat.data) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
}

inline void print_shape(const std::string& tensor_name, const std::vector<size_t>& shape) {
  std::cout << tensor_name << ": ";
  for (auto iter : shape) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
}