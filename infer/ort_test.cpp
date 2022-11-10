#include "onnxruntime/core/session/onnxruntime_c_api.h"

#include <iostream>
#include <stdio.h>

int main() {
  std::cout << "ORT_API_VERSION: " << ORT_API_VERSION << std::endl;
  return 0;
}