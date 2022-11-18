#include <cuda_runtime_api.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/core_io.h>

#include <iostream>
#include <stdio.h>

int main() {
  cutlass::half_t x = 2.25_hf;
  std::cout << x << std::endl;
  return 0;
}

