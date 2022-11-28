#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#define CHECK_CUDNN(expression)                                                       \
  do {                                                                                \
    cudnnStatus_t status = (expression);                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                                             \
      printf("[%s:%d] Error: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

inline std::vector<float> generate_cpu_data(size_t size) {
  std::vector<float> ret(size, 1.0f);
  return ret;
}

inline void print_vector(const std::vector<float>& vec, size_t line_size, std::ostream& o_stream) {
  for (size_t i = 0; i < vec.size(); i++) {
    if (i % line_size == 0 && i != 0) {
      o_stream << "\n";
    }
    o_stream << vec[i] << " ";
  }
  return;
}

// input : (n, c, h, w)
// w     : (k, c, r, s)
// output: (n, k, h, w)
// pad = 1, r = s = 3, so that output(h, w) == input(h, w)

int main() {
  cudnnHandle_t handle;
  CHECK_CUDNN(cudnnCreate(&handle));

  int64_t n = 1, c = 1, h = 16, w = 16;
  int64_t k = 1, r = 3, s = 3;

  // set input descriptor
  cudnnBackendDescriptor_t xDesc;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc));
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
  int64_t x_size = n * c * h * w;
  int64_t xDim[] = {n, c, h, w};
  int64_t xStr[] = {c * h * w, h * w, w, 1};
  int64_t xUi = 'x';
  int64_t alignment = 4;
  CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, xDim));
  CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, xStr));
  CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &xUi));
  CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
  CHECK_CUDNN(cudnnBackendFinalize(xDesc));

  // set filter descriptor
  cudnnBackendDescriptor_t wDesc;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &wDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
  int64_t w_size = k * c * r * s;
  int64_t wDim[] = {k, c, r, s};
  int64_t wStr[] = {c * r * s, r * s, s, 1};
  int64_t wUi = 'w';
  CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, wDim));
  CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, wStr));
  CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &wUi));
  CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
  CHECK_CUDNN(cudnnBackendFinalize(wDesc));

  // set output descriptor
  cudnnBackendDescriptor_t yDesc;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
  int64_t yDim[] = {n, k, h, w};
  int64_t y_size = n * k * h * w;
  int64_t yStr[] = {k * h * w, h * w, w, 1};
  int64_t yUi = 'y';
  CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, yDim));
  CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, yStr));
  CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &yUi));
  CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
  CHECK_CUDNN(cudnnBackendFinalize(yDesc));

  // set convolution descriptor
  cudnnBackendDescriptor_t cDesc;
  int64_t nbDims = 2;
  cudnnDataType_t compType = CUDNN_DATA_FLOAT;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  int64_t pad[] = {1, 1};
  int64_t filterStr[] = {1, 1};
  int64_t dilation[] = {1, 1};

  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &cDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &nbDims));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compType));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, nbDims, pad));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, nbDims, pad));
  CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, nbDims, dilation));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, nbDims, filterStr));
  CHECK_CUDNN(cudnnBackendFinalize(cDesc));

  // set convolution operator
  cudnnBackendDescriptor_t fprop;
  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &fprop));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1, &xDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1, &wDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1, &yDesc));
  CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cDesc));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  CHECK_CUDNN(cudnnBackendFinalize(fprop));

  // set graph descriptor
  int len = 1;
  cudnnBackendDescriptor_t ops[] = {fprop};
  cudnnBackendDescriptor_t op_graph;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, len, &fprop));
  CHECK_CUDNN(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
  CHECK_CUDNN(cudnnBackendFinalize(op_graph));

  int64_t support_engine_count = -1;
  CHECK_CUDNN(cudnnBackendGetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1,
                                       nullptr, &support_engine_count));
  std::cout << "support engine count: " << support_engine_count << std::endl;

  // set engine descriptor
  // cudnnBackendDescriptor_t engine;
  // CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
  // CHECK_CUDNN(
  //     cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
  //     &op_graph));
  // int64_t gidx = 0;
  // CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gidx));
  // CHECK_CUDNN(cudnnBackendFinalize(engine));

  // for search config
  cudnnBackendDescriptor_t heuristic_searcher;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristic_searcher));
  CHECK_CUDNN(cudnnBackendSetAttribute(heuristic_searcher, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
  cudnnBackendHeurMode_t search_mode = CUDNN_HEUR_MODE_A;
  CHECK_CUDNN(
      cudnnBackendSetAttribute(heuristic_searcher, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &search_mode));
  CHECK_CUDNN(cudnnBackendFinalize(heuristic_searcher));

  // set engine config descriptor
  // std::vector<cudnnBackendDescriptor_t> engcfgs(support_engine_count);
  // for (size_t i = 0; i < engcfgs.size(); i++) {
  //   CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, engcfgs.data() + i));
  //   CHECK_CUDNN(
  //       cudnnBackendSetAttribute(engcfgs[i], CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
  //       &engine));
  // }

  cudnnBackendDescriptor_t engcfg1;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engcfg1));
  // CHECK_CUDNN(
  //     cudnnBackendSetAttribute(engcfg1, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine));
  // CHECK_CUDNN(cudnnBackendFinalize(engcfg1));
  int64_t config_count = 1;
  int64_t return_config_count = -1;
  CHECK_CUDNN(cudnnBackendGetAttribute(heuristic_searcher, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       config_count, &return_config_count, &engcfg1));
  // CHECK_CUDNN(cudnnBackendFinalize(engcfg1));

  // set plan descriptor
  cudnnBackendDescriptor_t plan;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
  CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
                                       &engcfg1));
  CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
  CHECK_CUDNN(cudnnBackendFinalize(plan));
  int64_t workspaceSize;
  int64_t return_count;
  CHECK_CUDNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1,
                                       &return_count, &workspaceSize));

  // allooc device memory
  void* xData = nullptr;
  void* wData = nullptr;
  void* yData = nullptr;
  void* workspace = nullptr;
  cudaMalloc(&xData, sizeof(float) * x_size);
  cudaMalloc(&wData, sizeof(float) * w_size);
  cudaMalloc(&yData, sizeof(float) * y_size);
  cudaMalloc(&workspace, workspaceSize);
  auto x_vec = generate_cpu_data(x_size);
  auto w_vec = generate_cpu_data(w_size);
  cudaMemcpy(xData, x_vec.data(), sizeof(float) * x_size, cudaMemcpyHostToDevice);
  cudaMemcpy(wData, w_vec.data(), sizeof(float) * w_size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // set data pack descriptor
  void* dev_ptrs[3] = {xData, wData, yData};  // device pointer
  int64_t uids[3] = {'x', 'w', 'y'};

  cudnnBackendDescriptor_t varpack;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, dev_ptrs));
  CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids));
  CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace));
  CHECK_CUDNN(cudnnBackendFinalize(varpack));

  // exec
  CHECK_CUDNN(cudnnBackendExecute(handle, plan, varpack));

  cudaDeviceSynchronize();

  std::vector<float> y_vec(y_size, 3.0f);
  cudaMemcpy(y_vec.data(), yData, sizeof(float) * y_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  std::fstream x_fs("x.txt", std::ios::out | std::ios::trunc);
  std::fstream y_fs("y.txt", std::ios::out | std::ios::trunc);
  std::fstream w_fs("w.txt", std::ios::out | std::ios::trunc);
  print_vector(y_vec, w, y_fs);
  print_vector(w_vec, s, w_fs);
  print_vector(x_vec, w, x_fs);

  cudaFree(xData);
  cudaFree(yData);
  cudaFree(wData);
  cudaFree(workspace);

  CHECK_CUDNN(cudnnDestroy(handle));
  return 0;
}