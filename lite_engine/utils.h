#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <cudnn.h>

#define CHECK_CUDNN(expression)                                                       \
  do {                                                                                \
    cudnnStatus_t status = (expression);                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                                             \
      printf("[%s:%d] Error: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

inline std::vector<float> generate_cpu_data(size_t size) {
  std::vector<float> ret(size, 1.0f);
  return ret;
}

inline double now_ms() {
  struct timespec res;
  clock_gettime(CLOCK_REALTIME, &res);
  return 1000.0 * res.tv_sec + (double)res.tv_nsec / 1e6;
}