#include "conv_transpose.h"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include "utils.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

ConvTranspose::ConvTranspose() {
  CHECK_CUDNN(cudnnCreate(&handle_));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc_));
  CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc_));
  CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc_));
  CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));
  algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  return;
}

ConvTranspose::~ConvTranspose() {
  CHECK_CUDNN(cudnnDestroy(handle_));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc_));
  CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc_));
  CHECK_CUDNN(cudnnDestroyFilterDescriptor(weight_desc_));
  CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  return;
}

// input_shape: NCHW, weight_shape: CBhw , output_shape: NBHW
std::vector<size_t> ConvTranspose::InferShape(const std::vector<size_t>& input_shape, const std::vector<size_t>& weight_shape,
                               const ConvTransposeParam& param) const {
  // output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] +
  // ((kernel_shape[i] - 1) * dilations[i] + 1)
  // - pads[start_i] - pads[end_i]
  std::vector<size_t> output_shape(input_shape.size());
  output_shape[0] = input_shape[0];
  output_shape[1] = weight_shape[1];

  for (int i = 2; i < input_shape.size(); i++) {
    int axis = i - 2;
    output_shape[i] = param.strides[axis] * (input_shape[i] - 1) + param.output_padding[axis] +
                      ((weight_shape[i] - 1) * param.dilations[axis] + 1) - 2 * param.pads[axis];
  }
  return output_shape;
}

void ConvTranspose::Conv2dTranspose(const Tensor& input, const Tensor& weight, const ConvTransposeParam& param,
                                    Tensor& output) {
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.shape[0],
                                         input.shape[1], input.shape[2], input.shape[3]));
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.shape[0],
                                         output.shape[1], output.shape[2], output.shape[3]));
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weight.shape[0],
                                         weight.shape[1], weight.shape[2], weight.shape[3]));
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc_, param.pads[0], param.pads[1], param.strides[0],
                                              param.strides[1], param.dilations[0], param.dilations[1],
                                              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  size_t workspace_bytes = 0;
  void* workspace_data;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, weight_desc_, input_desc_, conv_desc_, output_desc_,
                                                           algo_, &workspace_bytes));

  if (workspace_bytes > 0) {
    cudaMalloc(&workspace_data, workspace_bytes);
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionBackwardData(handle_, &alpha, weight_desc_, (void*)weight.data, input_desc_,
                                           (void*)input.data, conv_desc_, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr, 0,
                                           &beta, output_desc_, (void*)output.data));
  if (workspace_bytes > 0) {
    cudaFree(workspace_data);
  }
  return;
}
