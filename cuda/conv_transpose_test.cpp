#include "conv_transpose.h"
#include "utils.h"

void test1() {
  Matrix input;
  input.shape = {1, 512, 14, 14};
  input.data = read_data(1 * 512 * 14 * 14, "./data/input.txt");

  Matrix weight;
  weight.shape = {512, 256, 3, 3};
  weight.data = read_data(512 * 256 * 3 * 3, "./data/w.txt");

  ConvTransposeParam param;
  param.dilations = {1, 1};
  param.groups = 1;
  param.output_padding = {0, 0};
  param.pads = {0, 0};
  param.strides = {1, 1};

  Tensor input_gpu;
  input_gpu.shape = input.shape;
  print_shape("input", input_gpu.shape);
  cudaMalloc((void**)&input_gpu.data, shape2size(input.shape) * sizeof(float));
  cudaMemcpy(input_gpu.data, input.data.data(), shape2size(input.shape) * sizeof(float), cudaMemcpyHostToDevice);

  Tensor weight_gpu;
  weight_gpu.shape = weight.shape;
  print_shape("weight", weight_gpu.shape);
  cudaMalloc((void**)&weight_gpu.data, shape2size(weight_gpu.shape) * sizeof(float));
  cudaMemcpy(weight_gpu.data, weight.data.data(), shape2size(weight_gpu.shape) * sizeof(float), cudaMemcpyHostToDevice);

  Tensor output_gpu;
  ConvTranspose conv_transpose;
  output_gpu.shape = conv_transpose.InferShape(input_gpu.shape, weight_gpu.shape, param);
  print_shape("output", output_gpu.shape);
  cudaMalloc((void**)&output_gpu.data, shape2size(output_gpu.shape) * sizeof(float));

  conv_transpose.Conv2dTranspose(input_gpu, weight_gpu, param, output_gpu);

  Matrix output;
  output.shape = output_gpu.shape;
  output.data.resize(shape2size(output_gpu.shape));
  cudaMemcpy(output.data.data(), output_gpu.data, shape2size(output_gpu.shape) * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "out: ";
  for (auto iter : output.shape) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
  write_data(output.data, shape2size(output.shape), "./output.my.txt");

  cudaFree(input_gpu.data);
  cudaFree(weight_gpu.data);
  cudaFree(output_gpu.data);
  return;
}

int main() {
  test1();
  return 0;
}