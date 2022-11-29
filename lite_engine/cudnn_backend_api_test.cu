#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <fstream>
#include <initializer_list>
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

inline void print_dims(int64_t* dims) {
  std::cout << *dims << "," << *(dims + 1) << "," << *(dims + 2) << "," << *(dims + 3);
  return;
}

// input dims must be (n, c, h, w), returned TensorDescriptor must be binded to a pointer with NHWC layout memory.
cudnnBackendDescriptor_t GetTensorDescriptorNHWC(const std::initializer_list<int64_t>& dims, int64_t id,
                                                 bool is_virtual = false) {
  if (dims.size() != 4) {
    printf("[Error] [%s:%d] cudnn Backend API only support dims == 4\n", __FILE__, __LINE__);
    exit(1);
  }
  cudnnBackendDescriptor_t tensor_desc;
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  int64_t alignment = 4;
  int64_t* dims_ptr = nullptr;
#if defined(__cplusplus) && __cplusplus >= 201703L
  dims_ptr = dims.data();
#else
  std::vector<int64_t> dims_vec(dims.begin(), dims.end());
  dims_ptr = dims_vec.data();
#endif
  std::vector<int64_t> strides(dims.begin(), dims.end());

  // NCHW format, cudnn conv/conv_transpose fusion only support layout == NHWC
  strides[0] = strides[1] * strides[2] * strides[3];
  strides[1] = strides[2] * strides[3];
  strides[2] = strides[3];
  strides[3] = 1;

  // NHWC
  std::vector<int64_t> dims_vec_for_nhwc(dims.begin(), dims.end());
  // strides[0] = dims_vec_for_nhwc[1] * dims_vec_for_nhwc[2] * dims_vec_for_nhwc[3];
  // strides[1] = 1;
  // strides[2] = dims_vec_for_nhwc[1] * dims_vec_for_nhwc[3];
  // strides[3] = dims_vec_for_nhwc[1];

  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(tensor_desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
  CHECK_CUDNN(cudnnBackendSetAttribute(tensor_desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims_ptr));
  CHECK_CUDNN(cudnnBackendSetAttribute(tensor_desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides.data()));
  CHECK_CUDNN(cudnnBackendSetAttribute(tensor_desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id));
  CHECK_CUDNN(cudnnBackendSetAttribute(tensor_desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
  if (is_virtual) {
    CHECK_CUDNN(
        cudnnBackendSetAttribute(tensor_desc, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual));
  }
  CHECK_CUDNN(cudnnBackendFinalize(tensor_desc));
  std::cout << (char)id << " dims:";
  print_dims(dims_ptr);
  std::cout << " strides:";
  print_dims(strides.data());
  if (is_virtual) {
    std::cout << " Virtual";
  }
  std::cout << std::endl;

  return tensor_desc;
}

cudnnBackendDescriptor_t GetConvOp(cudnnBackendDescriptor_t input_desc, cudnnBackendDescriptor_t w_desc,
                                   cudnnBackendDescriptor_t output_desc, const std::initializer_list<int64_t>& pads,
                                   const std::initializer_list<int64_t>& strides) {
  if (pads.size() != 2 || strides.size() != 2) {
    printf("[Error] [%s:%d] cudnn Backend API only support pads.dim == 2 && strides.dim == 2\n", __FILE__, __LINE__);
    exit(1);
  }

  // set convolution descriptor
  cudnnBackendDescriptor_t conv_desc;
  int64_t conv_dim = 2;
  cudnnDataType_t conv_dtype = CUDNN_DATA_FLOAT;
  cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;
  int64_t dilation[] = {1, 1};

  int64_t* pads_ptr = nullptr;
  int64_t* strides_ptr = nullptr;
#if defined(__cplusplus) && __cplusplus >= 201703L
  pads_ptr = pads.data();
  strides_ptr = strides.data();
#else
  std::vector<int64_t> pads_vec(pads.begin(), pads.end());
  pads_ptr = pads_vec.data();
  std::vector<int64_t> strides_vec(strides.begin(), strides.end());
  strides_ptr = strides_vec.data();
#endif

  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &conv_dim));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &conv_dtype));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1,
                                       &conv_mode));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, conv_dim, pads_ptr));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, conv_dim, pads_ptr));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, conv_dim, dilation));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, conv_dim,
                                       strides_ptr));
  CHECK_CUDNN(cudnnBackendFinalize(conv_desc));

  // set convolution fprop operation
  cudnnBackendDescriptor_t conv_op;
  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &conv_op));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_desc));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_op, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
  CHECK_CUDNN(cudnnBackendFinalize(conv_op));
  return conv_op;
}

cudnnBackendDescriptor_t GetConvTransposeOp(cudnnBackendDescriptor_t input_desc, cudnnBackendDescriptor_t w_desc,
                                            cudnnBackendDescriptor_t output_desc,
                                            const std::initializer_list<int64_t>& pads,
                                            const std::initializer_list<int64_t>& strides) {
  if (pads.size() != 2 || strides.size() != 2) {
    printf("[Error] [%s:%d] cudnn Backend API only support pads.dim == 2 && strides.dim == 2\n", __FILE__, __LINE__);
    exit(1);
  }

  // set convolution descriptor
  cudnnBackendDescriptor_t conv_desc;
  int64_t conv_dim = 2;
  cudnnDataType_t conv_dtype = CUDNN_DATA_FLOAT;
  cudnnConvolutionMode_t conv_mode = CUDNN_CROSS_CORRELATION;
  int64_t dilation[] = {1, 1};

  int64_t* pads_ptr = nullptr;
  int64_t* strides_ptr = nullptr;
#if defined(__cplusplus) && __cplusplus >= 201703L
  pads_ptr = pads.data();
  strides_ptr = strides.data();
#else
  std::vector<int64_t> pads_vec(pads.begin(), pads.end());
  pads_ptr = pads_vec.data();
  std::vector<int64_t> strides_vec(strides.begin(), strides.end());
  strides_ptr = strides_vec.data();
#endif

  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &conv_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &conv_dim));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &conv_dtype));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1,
                                       &conv_mode));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, conv_dim, pads_ptr));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, conv_dim, pads_ptr));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, conv_dim, dilation));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_desc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, conv_dim,
                                       strides_ptr));
  CHECK_CUDNN(cudnnBackendFinalize(conv_desc));

  // set convolution dgrad operation
  cudnnBackendDescriptor_t conv_transpose_op;
  float alpha = 1.0f;
  float beta = 0.0f;
  CHECK_CUDNN(
      cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR, &conv_transpose_op));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_transpose_op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_transpose_op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_transpose_op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_transpose_op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &conv_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_transpose_op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                                       CUDNN_TYPE_FLOAT, 1, &alpha));
  CHECK_CUDNN(cudnnBackendSetAttribute(conv_transpose_op, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                                       CUDNN_TYPE_FLOAT, 1, &beta));
  CHECK_CUDNN(cudnnBackendFinalize(conv_transpose_op));
  return conv_transpose_op;
}

cudnnBackendDescriptor_t GetReluOp(cudnnBackendDescriptor_t input_desc, cudnnBackendDescriptor_t output_desc) {
  // set relu descriptor
  cudnnBackendDescriptor_t relu_desc;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &relu_desc));
  cudnnPointwiseMode_t relu_mode = CUDNN_POINTWISE_RELU_FWD;
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  CHECK_CUDNN(cudnnBackendSetAttribute(relu_desc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &relu_mode));
  CHECK_CUDNN(cudnnBackendSetAttribute(relu_desc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
  CHECK_CUDNN(cudnnBackendFinalize(relu_desc));

  // set relu operation
  cudnnBackendDescriptor_t relu_op;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &relu_op));
  CHECK_CUDNN(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &relu_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
                                       &input_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(relu_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
                                       &output_desc));
  CHECK_CUDNN(cudnnBackendFinalize(relu_op));
  return relu_op;
}

cudnnBackendDescriptor_t GetPointwiseOp(const cudnnBackendDescriptor_t& first_input_desc,
                                        const cudnnBackendDescriptor_t& second_input_desc,
                                        const cudnnBackendDescriptor_t& output_desc, cudnnPointwiseMode_t mode) {
  // set pointwise descriptor
  if (mode != CUDNN_POINTWISE_ADD && mode != CUDNN_POINTWISE_MUL) {
    printf("[Error] cudnn Backend API pointwise only support mul & add\n");
    exit(1);
  }
  cudnnBackendDescriptor_t pointwise_desc;
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pointwise_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(pointwise_desc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(pointwise_desc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
  CHECK_CUDNN(cudnnBackendFinalize(pointwise_desc));
  // set pointwise op
  cudnnBackendDescriptor_t pointwise_op;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &pointwise_op));
  CHECK_CUDNN(cudnnBackendSetAttribute(pointwise_op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pointwise_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(pointwise_op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &first_input_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(pointwise_op, CUDNN_ATTR_OPERATION_POINTWISE_BDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &second_input_desc));
  CHECK_CUDNN(cudnnBackendSetAttribute(pointwise_op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_desc));
  CHECK_CUDNN(cudnnBackendFinalize(pointwise_op));
  return pointwise_op;
}

cudnnBackendDescriptor_t GetGraph(cudnnHandle_t handle, cudnnBackendDescriptor_t* ops, int64_t len) {
  cudnnBackendDescriptor_t op_graph;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, len, ops));
  CHECK_CUDNN(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
  CHECK_CUDNN(cudnnBackendFinalize(op_graph));

  // debug info
  int64_t graph_support_engine_count = -1;
  CHECK_CUDNN(cudnnBackendGetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT, CUDNN_TYPE_INT64, 1,
                                       nullptr, &graph_support_engine_count));
  std::cout << "cudnn graph support engine count: " << graph_support_engine_count << std::endl;

  return op_graph;
}

cudnnBackendDescriptor_t GetEngineSearcher(cudnnBackendDescriptor_t op_graph) {
  cudnnBackendDescriptor_t heuristic_searcher;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristic_searcher));
  CHECK_CUDNN(cudnnBackendSetAttribute(heuristic_searcher, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
  cudnnBackendHeurMode_t search_mode = CUDNN_HEUR_MODE_FALLBACK;
  CHECK_CUDNN(
      cudnnBackendSetAttribute(heuristic_searcher, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &search_mode));
  CHECK_CUDNN(cudnnBackendFinalize(heuristic_searcher));
  return heuristic_searcher;
}

cudnnBackendDescriptor_t FindBestEngineConfig(cudnnBackendDescriptor_t searcher) {
  cudnnBackendDescriptor_t best_engine_config;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &best_engine_config));
  int64_t config_count = 1;
  int64_t return_config_count = -1;
  CHECK_CUDNN(cudnnBackendGetAttribute(searcher, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                       config_count, &return_config_count, &best_engine_config));
  return best_engine_config;
}

cudnnBackendDescriptor_t GetPlan(cudnnHandle_t handle, cudnnBackendDescriptor_t config, int64_t& workspace_bytes) {
  cudnnBackendDescriptor_t plan;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
  CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1,
                                       &config));
  CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle));
  CHECK_CUDNN(cudnnBackendFinalize(plan));
  int64_t return_count;
  CHECK_CUDNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1,
                                       &return_count, &workspace_bytes));
  return plan;
}

// input : (n, c, h, w)
// w     : (k, c, r, s)
// output: (n, k, h, w)
// pad = 1, r = s = 3, so that output(h, w) == input(h, w)
// c is input channel, k is output channel

int main() {
  std::cout << "CUDNN VERSION: " << CUDNN_MAJOR << "." << CUDNN_MINOR << std::endl;
  cudnnHandle_t handle;
  CHECK_CUDNN(cudnnCreate(&handle));

  int64_t n = 1, c = 512, h = 16, w = 16;
  int64_t k = 512, w_c = 256, r = 3, s = 3;
  int64_t out_c = w_c, out_h = 32, out_w = 32;
  int64_t pad = 1;
  // int64_t out_pad = 1;
  int64_t stride = 2;

  // set input descriptor
  cudnnBackendDescriptor_t input_desc = GetTensorDescriptorNHWC({n, c, h, w}, 'x');
  int64_t input_size = n * c * h * w;

  // set filter descriptor
  cudnnBackendDescriptor_t conv_weight_desc = GetTensorDescriptorNHWC({k, w_c, r, s}, 'w');
  int64_t conv_weight_size = k * w_c * r * s;

  // set output descriptor
  cudnnBackendDescriptor_t conv_output_desc = GetTensorDescriptorNHWC({n, out_c, out_h, out_w}, 'C', true);
  int64_t conv_output_size = n * out_c * out_h * out_w;

  cudnnBackendDescriptor_t z_desc = GetTensorDescriptorNHWC({n, out_c, out_h, out_w}, 'z');
  int64_t z_size = n * out_c * out_h * out_w;

  cudnnBackendDescriptor_t add_op1_output_desc = GetTensorDescriptorNHWC({n, out_c, out_h, out_w}, 'A');

  cudnnBackendDescriptor_t bias_desc = GetTensorDescriptorNHWC({1, out_c, 1, 1}, 'b');
  int64_t b_size = 1 * out_c * 1 * 1;

  cudnnBackendDescriptor_t add_op2_output_desc = GetTensorDescriptorNHWC({n, out_c, out_h, out_w}, 'B');

  cudnnBackendDescriptor_t output_desc = GetTensorDescriptorNHWC({n, out_c, out_h, out_w}, 'y');

  // set convolution operator
  // cudnnBackendDescriptor_t fprop =
  //     GetConvOp(conv_output_desc, conv_weight_desc, input_desc, {pad, pad}, {stride, stride});
  cudnnBackendDescriptor_t dgrad =
      GetConvTransposeOp(input_desc, conv_weight_desc, conv_output_desc, {pad, pad}, {stride, stride});

  cudnnBackendDescriptor_t add_op1 = GetPointwiseOp(conv_output_desc, z_desc, add_op1_output_desc, CUDNN_POINTWISE_ADD);

  cudnnBackendDescriptor_t add_op2 =
      GetPointwiseOp(add_op1_output_desc, bias_desc, add_op2_output_desc, CUDNN_POINTWISE_ADD);

  // set relu operator
  cudnnBackendDescriptor_t relu = GetReluOp(conv_output_desc, output_desc);

  // set ConvTranspose, use ConvolutionBackword for impl, dy is x, w is w, dx is y
  // cudnnBackendDescriptor_t dgrad;
  // CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
  // &dgrad)); CHECK_CUDNN(cudnnBackendSetAttribute(dgrad, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
  //                                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc));
  // CHECK_CUDNN(cudnnBackendSetAttribute(dgrad, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
  //                                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc));
  // CHECK_CUDNN(cudnnBackendSetAttribute(dgrad, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
  //                                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &afterAddDesc));
  // CHECK_CUDNN(cudnnBackendSetAttribute(dgrad, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC,
  //                                      CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cDesc));
  // CHECK_CUDNN(
  //     cudnnBackendSetAttribute(dgrad, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA, CUDNN_TYPE_FLOAT, 1,
  //     &alpha));
  // CHECK_CUDNN(
  //     cudnnBackendSetAttribute(dgrad, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA, CUDNN_TYPE_FLOAT, 1,
  //     &beta));
  // CHECK_CUDNN(cudnnBackendFinalize(dgrad));

  // set graph descriptor
  int64_t len = 2;
  cudnnBackendDescriptor_t ops[] = {dgrad, relu};
  cudnnBackendDescriptor_t op_graph = GetGraph(handle, ops, len);

  // for search config
  cudnnBackendDescriptor_t heuristic_searcher = GetEngineSearcher(op_graph);

  cudnnBackendDescriptor_t best_config = FindBestEngineConfig(heuristic_searcher);

  // set plan descriptor
  int64_t workspaceSize;
  cudnnBackendDescriptor_t plan = GetPlan(handle, best_config, workspaceSize);

  // allooc device memory
  void* xData = nullptr;
  void* wData = nullptr;
  void* yData = nullptr;
  void* zData = nullptr;
  void* bData = nullptr;
  void* workspace = nullptr;
  cudaMalloc(&xData, sizeof(float) * input_size);
  cudaMalloc(&wData, sizeof(float) * conv_weight_size);
  cudaMalloc(&yData, sizeof(float) * conv_output_size);
  cudaMalloc(&zData, sizeof(float) * z_size);
  cudaMalloc(&bData, sizeof(float) * b_size);
  cudaMalloc(&workspace, workspaceSize);
  auto x_vec = generate_cpu_data(input_size);
  auto w_vec = generate_cpu_data(conv_weight_size);
  auto z_vec = generate_cpu_data(z_size);
  auto b_vec = generate_cpu_data(b_size);
  cudaMemcpy(xData, x_vec.data(), sizeof(float) * input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(wData, w_vec.data(), sizeof(float) * conv_weight_size, cudaMemcpyHostToDevice);
  cudaMemcpy(zData, z_vec.data(), sizeof(float) * z_size, cudaMemcpyHostToDevice);
  cudaMemcpy(bData, b_vec.data(), sizeof(float) * b_size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // set data pack descriptor
  int data_count = 5;
  void* dev_ptrs[data_count] = {xData, wData, yData, zData, bData};  // device pointer
  int64_t uids[data_count] = {'x', 'w', 'y', 'z', 'b'};

  cudnnBackendDescriptor_t varpack;
  CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack));
  CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, data_count,
                                       dev_ptrs));
  CHECK_CUDNN(
      cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, data_count, uids));
  CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &workspace));
  CHECK_CUDNN(cudnnBackendFinalize(varpack));

  // exec
  CHECK_CUDNN(cudnnBackendExecute(handle, plan, varpack));

  cudaDeviceSynchronize();

  std::vector<float> y_vec(conv_output_size, -1.0f);
  cudaMemcpy(y_vec.data(), yData, sizeof(float) * conv_output_size, cudaMemcpyDeviceToHost);
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