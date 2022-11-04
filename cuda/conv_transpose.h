#pragma once

#include <cudnn.h>
#include <vector>
#include "utils.h"

struct ConvTransposeParam {
  std::vector<int> dilations;
  int groups;
  std::vector<int> output_padding;
  std::vector<int> pads;
  std::vector<int> strides;
};

class ConvTranspose final {
 public:
  ConvTranspose();
  ~ConvTranspose();
  ConvTranspose(ConvTranspose&& other) = delete;
  ConvTranspose(const ConvTranspose& other) = delete;
  ConvTranspose operator=(ConvTranspose&& other) = delete;
  ConvTranspose operator=(const ConvTranspose& other) = delete;

  void InferShape(const std::vector<size_t>& input_shape, const std::vector<size_t>& weight_shape, const ConvTransposeParam& param,
                  std::vector<size_t>& output_shape) const;
  void FindBestAlgorithms(const Tensor& input, const Tensor& weight, const ConvTransposeParam& param, Tensor& output);
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
  size_t workspace_bytes_;
  void* workspace_data_;
};