import sys
import onnx
from onnx import AttributeProto, TensorProto, GraphProto 

def set_batch(model, batch):
  graph = model.graph
  input = onnx.helper.make_tensor_value_info(name="img_in", elem_type=TensorProto.FLOAT, shape=[batch, 3, 512, 512])
  output1 = onnx.helper.make_tensor_value_info(name="img_out", elem_type=TensorProto.FLOAT, shape=[batch, 3, 512, 512])
  output2 = onnx.helper.make_tensor_value_info(name="mask", elem_type=TensorProto.FLOAT, shape=[batch, 1, 512, 512])
  new_graph = onnx.helper.make_graph(nodes=graph.node, name=graph.name, inputs=[input], outputs=[output1, output2], initializer=graph.initializer, value_info=graph.value_info)
  new_model = onnx.helper.make_model(new_graph, producer_name="tangjicheng")
  new_model = onnx.shape_inference.infer_shapes(new_model)
  onnx.checker.check_model(new_model)
  return new_model

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("usage: python change_input_batch.py model.onnx")
    sys.exit(1)
  model = onnx.load(sys.argv[1])
  for i in [1, 4, 8, 16, 32, 100]:
    new_model_name = "batch_" + str(i) + ".onnx"
    new_model = set_batch(model, i)
    onnx.save(new_model, new_model_name)
