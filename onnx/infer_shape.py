import sys
import onnx
from onnx import shape_inference

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("usage: python infer_shape.py input.onnx output.onnx")
    sys.exit(1)
  input_name = sys.argv[1]
  output_name = sys.argv[2]
  input_model = onnx.load(input_name)
  output_model = onnx.shape_inference.infer_shapes(input_model)
  onnx.save(output_model, output_name)
  # onnx.save(onnx.shape_inference.infer_shapes(onnx.load(input_name)), output_name)
