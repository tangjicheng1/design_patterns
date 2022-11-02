#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <stdio.h>
#include <iostream>
#include <vector>

#define CHECK_CUDNN(expression)                                                       \
  do {                                                                                \
    cudnnStatus_t status = (expression);                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                                             \
      printf("[%s:%d] Error: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

struct matrix {
  std::vector<float> data;
  std::vector<int> shape;
};

struct conv_param {
  std::vector<int> dilations;
  int groups;
  std::vector<int> output_padding;
  std::vector<int> pads;
  std::vector<int> strides;
};

int shape2size(const std::vector<int>& shape) {
  int ret = 1;
  for (auto iter : shape) {
    ret *= iter;
  }
  return ret;
}

// input_shape: NCHW, kernel_shape: BCHW , output_shape: NBHW
std::vector<int> infer_shape(const std::vector<int>& input_shape, const std::vector<int>& kernel_shape,
                             const conv_param& param) {
  std::vector<int> output_shape(4);
  // output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1)
  // - pads[start_i] - pads[end_i]
  output_shape[0] = input_shape[0];
  output_shape[1] = kernel_shape[0];

  for (int i = 2; i < input_shape.size(); i++) {
    int axis = i - 2;
    output_shape[i] = param.strides[axis] * (input_shape[i] - 1) + param.output_padding[axis] +
                      ((kernel_shape[i] - 1) * param.dilations[axis] + 1) - 2 * param.pads[axis];
  }

  return output_shape;
}

void conv_transpose(const matrix& input, const matrix& w, const conv_param& param, matrix& output) {
  output.shape = infer_shape(input.shape, w.shape, param);
  int output_size = shape2size(output.shape);
  output.data.resize(output_size);

  cudnnHandle_t cudnn;
  CHECK_CUDNN(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.shape[0],
                                         input.shape[1], input.shape[2], input.shape[3]));

  cudnnTensorDescriptor_t output_desc;
  cudnnCreateTensorDescriptor(&output_desc);
  CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, output.shape[0],
                                         output.shape[1], output.shape[2], output.shape[3]));

  cudnnFilterDescriptor_t w_desc;
  cudnnCreateFilterDescriptor(&w_desc);
  CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w.shape[0], w.shape[1],
                                         w.shape[2], w.shape[3]));

  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, param.pads[0], param.pads[1], param.strides[0],
                                              param.strides[1], param.dilations[0], param.dilations[1], conv_mode,
                                              CUDNN_DATA_FLOAT));

  void* w_data = nullptr;
  cudaMalloc(&w_data, w.data.size() * sizeof(float));
  cudaMemcpy(w_data, w.data.data(), w.data.size() * sizeof(float), cudaMemcpyHostToDevice);

  void* input_data = nullptr;
  cudaMalloc(&input_data, input.data.size() * sizeof(float));
  cudaMemcpy(input_data, input.data.data(), input.data.size() * sizeof(float), cudaMemcpyHostToDevice);

  void* output_data = nullptr;
  cudaMalloc(&output_data, output_size * sizeof(float));

  cudnnConvolutionBwdDataAlgo_t algo;
  int algo_num = 0;
  cudnnConvolutionBwdDataAlgoPerf_t algo_result;
  cudnnFindConvolutionBackwardDataAlgorithm(cudnn, w_desc, input_desc, conv_desc, output_desc, 1, &algo_num,
                                            &algo_result);
  algo = algo_result.algo;
  size_t workspace_bytes = algo_result.memory;
  void* workspace_data = nullptr;
  cudaMalloc(&workspace_data, workspace_bytes);

  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn, &alpha, w_desc, w_data, input_desc, input_data, conv_desc, algo,
                                           workspace_data, workspace_bytes, &beta, output_desc, output_data));

  cudaMemcpy(output.data.data(), output_data, output.data.size() * sizeof(float), cudaMemcpyDeviceToHost);

  CHECK_CUDNN(cudnnDestroy(cudnn));

  cudaFree(input_data);
  cudaFree(output_data);
  cudaFree(w_data);
  cudaFree(workspace_data);
  return;
}

void gen_data(std::vector<float>& data, int size) {
  data.resize(size);
  for (int i = 0; i < size; i++) {
    data[i] = i + 1;
  }
  return;
}

void print(const matrix& mat) {
  std::cout << "shape: ";
  for (auto iter : mat.shape) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
  std::cout << "data: ";
  for (auto iter : mat.data) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
}

void test1() {
  matrix input;
  input.shape = {1, 1, 4, 4};
  gen_data(input.data, shape2size(input.shape));

  matrix kernel;
  kernel.shape = {1, 1, 3, 3};
  gen_data(kernel.data, shape2size(kernel.shape));

  conv_param param;
  param.dilations = {1, 1};
  param.groups = 1;
  param.output_padding = {0, 0};
  param.pads = {0, 0};
  param.strides = {1, 1};

  matrix output;
  conv_transpose(input, kernel, param, output);

  print(input);
  print(kernel);
  std::cout << "out:\n";
  print(output);
  return;
}

int main() {
  test1();
  return 0;
}