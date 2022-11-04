#pragma once

#include <cudnn.h>
#include <vector>
#include "utils.h"

struct ConvTransposeParam {
  std::vector<int> dilations;
  std::vector<int> output_padding;
  std::vector<int> pads;
  std::vector<int> strides;
  int groups;
};

class ConvTranspose final {
 public:
  ConvTranspose();
  ~ConvTranspose();
  ConvTranspose(ConvTranspose&& other) = delete;
  ConvTranspose(const ConvTranspose& other) = delete;
  ConvTranspose operator=(ConvTranspose&& other) = delete;
  ConvTranspose operator=(const ConvTranspose& other) = delete;

  std::vector<size_t> InferShape(const std::vector<size_t>& input_shape, const std::vector<size_t>& weight_shape,
                                 const ConvTransposeParam& param) const;
  void FindBestAlgorithms(const std::vector<size_t>& input, const std::vector<size_t>& weight, const ConvTransposeParam& param);
  void Conv1dTranspose(const Tensor& input, const Tensor& weight, const ConvTransposeParam& param, Tensor& output);
  void Conv2dTranspose(const Tensor& input, const Tensor& weight, const ConvTransposeParam& param, Tensor& output);
  void Conv3dTranspose(const Tensor& input, const Tensor& weight, const ConvTransposeParam& param, Tensor& output);

 private:
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t weight_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionBwdDataAlgo_t algo_;
};