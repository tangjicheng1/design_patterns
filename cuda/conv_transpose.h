#pragma once

#include <cudnn.h>
#include "utils.h"

class ConvTranspose final {
  public:
  ConvTranspose();
  ~ConvTranspose();
  ConvTranspose(ConvTranspose&& other) = delete;
  ConvTranspose(const ConvTranspose& other) = delete;
  ConvTranspose operator=(ConvTranspose&& other) = delete;
  ConvTranspose operator=(const ConvTranspose& other) = delete;

  void Init();
  void Init(const CudaVec& weight);

  private:
};