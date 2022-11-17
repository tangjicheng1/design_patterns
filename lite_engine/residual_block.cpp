#include "tensor.h"
#include "residual_block.h"
#include "utils.h"

#include <limits>
#include <iostream>

void ResidualBlock::init(const ResidualBlockParam& param) {
  CHECK_CUDNN(cudnnCreate(&handle_));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&z_desc_));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc_));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc_));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
  CHECK_CUDNN(cudnnCreateActivationDescriptor(&act_desc_));

  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, param.batch, param.in_channel, param.input_h, param.input_w));
  std::vector<int> output_shape = infer_shape(param);
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output_shape[0], output_shape[1], output_shape[2], output_shape[3]));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(z_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, param.batch, param.in_channel, param.input_h, param.input_w));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, param.out_channel, 1, 1));
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, param.in_channel, param.out_channel, param.kernel_h, param.kernel_w));
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc_, param.pad_h, param.pad_w, param.stride_h, param.stride_w, param.dilation_h, param.dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CHECK_CUDNN(cudnnSetActivationDescriptor(act_desc_, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, std::numeric_limits<double>::max()));

  AlgoParam algo_param = find_best_algo();
  algo_ = algo_param.algo;
  std::cout << algo_ << std::endl;
  workspace_bytes_ = algo_param.workspace_bytes;
  if (workspace_ != nullptr) {
    cudaFree(workspace_);
  }
  if (workspace_bytes_ > 0) {
    cudaMalloc(&workspace_, algo_param.workspace_bytes);
  }

  // TODO: init weight bias
  std::vector<int> weight_shape = {param.out_channel, param.in_channel, param.kernel_h, param.kernel_w};
  std::vector<int> bias_shape = {param.out_channel};
  weight_.resize(weight_shape);
  bias_.resize(bias_shape);
  weight_.copy_from_host(generate_cpu_data(shape2size(weight_shape)));
  bias_.copy_from_host(generate_cpu_data(shape2size(bias_shape)));

  return;
}

void ResidualBlock::exec(const std::vector<Tensor*>& inputs, std::vector<Tensor*>& outputs) {
  const void* input_ptr = inputs[0]->data();
  void* output_ptr = outputs[0]->data();
  algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t cur_workspace_bytes = 0;
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle_, input_desc_, weight_desc_, conv_desc_, output_desc_, algo_, &cur_workspace_bytes));
  std::cout << cur_workspace_bytes << std::endl;
  CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
      handle_, &alpha1_, input_desc_, input_ptr, weight_desc_, weight_.data(), conv_desc_, algo_, workspace_,
      workspace_bytes_, &alpha2_, z_desc_, input_ptr, bias_desc_, bias_.data(), act_desc_, output_desc_, output_ptr));
  return;
}

void ResidualBlock::fini() {
  CHECK_CUDNN(cudnnDestroy(handle_));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc_));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc_));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(z_desc_));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc_));
  CHECK_CUDNN(cudnnDestroyFilterDescriptor(weight_desc_));
  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  CHECK_CUDNN(cudnnDestroyActivationDescriptor(act_desc_));

  if (workspace_ != nullptr) {
    cudaFree(workspace_);
  }

  return;
}

AlgoParam ResidualBlock::find_best_algo() {
  AlgoParam ret;
  int req_count = 1;
  int return_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf;
  CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(handle_, input_desc_, weight_desc_, conv_desc_, output_desc_, req_count, &return_count, &perf));
  ret.algo = perf.algo;
  ret.workspace_bytes = perf.memory;
  return ret;
}

std::vector<int> ResidualBlock::infer_shape(const ResidualBlockParam& param) {
  std::vector<int> ret(4);
  ret[0] = param.batch;
  ret[1] = param.out_channel;
  ret[2] = (param.input_h - param.kernel_h + 2 * param.pad_h) / param.stride_h + 1;
  ret[3] = (param.input_w - param.kernel_w + 2 * param.pad_w) / param.stride_w + 1;
  return ret;
}