#pragma once
#include "tensor.h"
#include <cudnn.h>

struct ResidualBlockParam {
  int batch;
  int in_channel;
  int out_channel;
  int input_h;
  int input_w;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int pad_h;
  int pad_w;
  int dilation_h;
  int dilation_w;
};

struct AlgoParam {
  cudnnConvolutionFwdAlgo_t algo;
  size_t workspace_bytes;
};

class ResidualBlock {
public:
  void init(const ResidualBlockParam& param);
  void exec(const std::vector<Tensor*>& inputs, std::vector<Tensor*>& output);
  void fini();
private:
  AlgoParam find_best_algo();
  std::vector<int> infer_shape(const ResidualBlockParam& param);
private:
  Tensor weight_;
  Tensor bias_;
  cudnnHandle_t handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnTensorDescriptor_t z_desc_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t weight_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnActivationDescriptor_t act_desc_;
  cudnnConvolutionFwdAlgo_t algo_;
  size_t workspace_bytes_ = 0;
  void* workspace_ = nullptr;
  float alpha1_ = 1.0f;
  float alpha2_ = 1.0f;
};