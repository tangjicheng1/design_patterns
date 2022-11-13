import sys
import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto

def get_conv_model(batch, input_c, input_h, input_w, weight_k, weight_h, weight_w, is_with_bias):
  ''''
    生成包含1个conv算子的onnx模型
    输入: input_shape = (batch, input_c, input_h, input_w)
         weight_shape = (weight_k, input_c, weight_h, weight_w)
         is_with_bias: 是否含有bias权重
    输出: 包含一个conv算子的ModeProto
    说明: weight和bias数据随机生成, output_shape由input_shape和weight_shape自动推导。
  '''

  # set input & output
  input_n = batch
  input_shape = [input_n, input_c, input_h, input_w]
  X = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape=input_shape)
  Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape=["batch", "c", "h", "w"])

  # set weight as constant_node
  weight_c = input_c
  weight = np.random.rand(weight_k, weight_c, weight_h, weight_w)
  weight = np.array(weight, dtype=np.float32)
  weight_tensor = numpy_helper.from_array(weight)
  weight_node = helper.make_node("Constant", name="weight_node", inputs=[], outputs=["weight"], value=weight_tensor)

  # set bias as constant_node
  if is_with_bias:
    bias = np.random.rand(weight_k)
    bias = np.array(bias, dtype=np.float32)
    bias_tensor = numpy_helper.from_array(bias)
    bias_node = helper.make_node("Constant", name="bias_node", inputs=[], outputs=["bias"], value=bias_tensor)

  # set conv node
  if is_with_bias:
    conv_node = helper.make_node("Conv", inputs=["X", "weight", "bias"], outputs=[
                                 "Y"], kernel_shape=[weight_h, weight_w])
  else:
    conv_node = helper.make_node("Conv", inputs=["X", "weight"],
                                 outputs=["Y"], kernel_shape=[weight_h, weight_w])

  # set graph & generate model
  if is_with_bias:
    graph = helper.make_graph(nodes=[bias_node, weight_node, conv_node], name="single_conv_model", inputs=[X], outputs=[Y])
  else: 
    graph = helper.make_graph(nodes=[weight_node, conv_node], name="single_conv_model", inputs=[X], outputs=[Y])
  model = helper.make_model(graph, producer_name="tangjicheng")
  model = onnx.shape_inference.infer_shapes(model)
  onnx.checker.check_model(model)
  return model


def generate_conv_models(directory_name):
  '''
    生成一组conv_model, 存放在directory_name目录下
  '''
  if directory_name[-1] != '/':
    directory_name += '/'
  batch_list = [1, 4, 8, 16, 32]
  for b in batch_list:
    # model without bias
    model = get_conv_model(batch=b, input_c=3, input_h=512, input_w=512, weight_k=16, weight_h=3, weight_w=3, is_with_bias=False)
    onnx_filename = directory_name + "conv_" + str(b) + ".onnx"
    onnx.save(model, onnx_filename)
    # model with bias
    model_with_bias = get_conv_model(batch=b, input_c=3, input_h=512, input_w=512, weight_k=16, weight_h=3, weight_w=3, is_with_bias=True)
    onnx_with_bias_filename = directory_name + "conv_" + str(b) + "_bias.onnx"
    onnx.save(model_with_bias, onnx_with_bias_filename)

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("usage: python conv_gen.py directory")
    sys.exit(1)
  generate_conv_models(sys.argv[1])
