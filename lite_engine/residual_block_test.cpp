#include "residual_block.h"
#include "utils.h"

const int run_loop = 100;
const int count_begin = 50;
const double warmup_time = 10000.0;  // ms, == 10 second.

void test1() {
  ResidualBlockParam param;
  int batch = 1;
  param.batch = batch;
  param.in_channel = 16;
  param.out_channel = 16;
  param.input_h = 256;
  param.input_w = 256;
  param.kernel_h = 3;
  param.kernel_w = 3;
  param.stride_h = 1;
  param.stride_w = 1;
  param.pad_h = 1;
  param.pad_w = 1;
  param.dilation_h = 1;
  param.dilation_w = 1;

  ResidualBlock res;
  res.init(param);

  std::vector<int> input_shape = {batch, 16, 256, 256};
  std::vector<int> output_shape = {batch, 16, 256, 256};
  Tensor input(input_shape);
  Tensor output(output_shape);
  input.copy_from_host(generate_cpu_data(shape2size(input_shape)));
  std::vector<Tensor*> input_vec = {&input};
  std::vector<Tensor*> output_vec = {&output};

  double warmup_cost = 0.0;
  double sum_cost = 0.0;
  int count = 0;
  int i = 0;
  while (true) {
    cudaDeviceSynchronize();
    double start_ms = now_ms();
    res.exec(input_vec, output_vec);

    cudaDeviceSynchronize();
    double end_ms = now_ms();
    double cost_ms = end_ms - start_ms;

    // warm up
    double cur_cost = end_ms - start_ms;
    warmup_cost += cur_cost;
    if (warmup_cost < warmup_time) {
      continue;
    }

    i += 1;
    // printf("cost: %lf\n", cost_ms);
    if (i > count_begin) {
      count += 1;
      sum_cost += cost_ms;
    }

    if (i > run_loop) {
      break;
    }
  }

  if (count > 0) {
    printf("average cost: %lf\n", sum_cost / count);
  }

  res.fini();
  return;
}

int main() {
  test1();
  return 0;
}