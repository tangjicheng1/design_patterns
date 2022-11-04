#pragma once

#include <stddef.h>
#include <cudnn.h>

struct CudaVec {
  void* data;
  size_t size;
};

#define CHECK_CUDNN(expression)                                                       \
  do {                                                                                \
    cudnnStatus_t status = (expression);                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                                             \
      printf("[%s:%d] Error: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)