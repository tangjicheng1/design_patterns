#include <cudnn.h>

#include <cudnn_ops_infer.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

#define check(expression)                                                             \
  do {                                                                                \
    cudnnStatus_t status = (expression);                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                                             \
      printf("[%s:%d] Error: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

std::vector<float> gen_data(size_t size) {
  std::vector<float> ret(size);
  for (size_t i = 0; i < size; i++) {
    ret[i] = i + 1;
  }
  return ret;
}

int main() {
  cudnnHandle_t cudnn;
  check(cudnnCreate(&cudnn));
  cudnnTensorDescriptor_t input_tensor;
  int n = 1;
  int c = 1;
  int h = 4;
  int w = 4;

  int k1 = 1;
  int k2 = 1;
  int k3 = 3;
  int k4 = 3;

  check(cudnnCreateTensorDescriptor(&input_tensor));
  check(cudnnSetTensor4dDescriptor(input_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

  cudnnTensorDescriptor_t output_tensor;
  check(cudnnCreateTensorDescriptor(&output_tensor));
  check(cudnnSetTensor4dDescriptor(output_tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));

  cudnnFilterDescriptor_t kernel_tensor;
  check(cudnnCreateFilterDescriptor(&kernel_tensor));
  check(cudnnSetFilter4dDescriptor(kernel_tensor, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k1, k2, k3, k4));

  int pad_h = 1;
  int pad_w = 1;
  int stride_v = 1;
  int stride_h = 1;
  int dilation_h = 1;
  int dilation_w = 1;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  cudnnDataType_t conv_type = CUDNN_DATA_FLOAT;

  cudnnConvolutionDescriptor_t conv_dest;
  check(cudnnCreateConvolutionDescriptor(&conv_dest));
  check(cudnnSetConvolution2dDescriptor(conv_dest, pad_h, pad_w, stride_h, stride_v, dilation_h, dilation_w, mode,
                                        conv_type));

  cudnnConvolutionFwdAlgoPerf_t conv_algo;
  int algo_num;
  check(cudnnFindConvolutionForwardAlgorithm(cudnn, input_tensor, kernel_tensor, conv_dest, output_tensor, 1, &algo_num,
                                             &conv_algo));

  size_t workspace_bytes = 0;
  check(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_tensor, kernel_tensor, conv_dest, output_tensor,
                                                conv_algo.algo, &workspace_bytes));

  void* workspace_mem = nullptr;
  cudaMalloc(&workspace_mem, workspace_bytes);

  int input_size = n * c * h * w;
  int output_size = n * c * h * w;
  int kernel_size = k1 * k2 * k3 * k4;

  float* input_data = nullptr;
  cudaMalloc(&input_data, input_size * sizeof(float));
  float* output_data = nullptr;
  cudaMalloc(&output_data, output_size * sizeof(float));
  float* kernel_data = nullptr;
  cudaMalloc(&kernel_data, kernel_size * sizeof(float));
  cudaMemset(output_data, 0, output_size * sizeof(float));

  std::vector<float> input_cpu = gen_data(input_size);
  std::vector<float> kernel_cpu = gen_data(kernel_size);

  cudaMemcpy(input_data, input_cpu.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(kernel_data, kernel_cpu.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = 0.0f;
  check(cudnnConvolutionForward(cudnn, &alpha, input_tensor, input_data, kernel_tensor, kernel_data, conv_dest,
                                conv_algo.algo, workspace_mem, workspace_bytes, &beta, output_tensor, output_data));

  std::vector<float> output_cpu(output_size);
  cudaMemcpy(output_cpu.data(), output_data, output_size * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "output: \n";
  for (int i = 0; i < output_size; i++) {
    std::cout << output_cpu[i] << " ";
  }
  std::cout << std::endl;
  
  cudaFree(workspace_mem);
  cudaFree(input_data);
  cudaFree(output_data);
  cudaFree(kernel_data);
  return 0;
}