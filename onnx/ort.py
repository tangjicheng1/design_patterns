import numpy as np
import onnx
import onnxruntime
from onnxruntime import backend
from onnxruntime.transformers import optimizer
import time

sess_options = onnxruntime.SessionOptions()

# 是否开启profiling功能
# sess_options.enable_profiling = True

model_name = "../conv/conv_1.onnx"
session = onnxruntime.InferenceSession(model_name, sess_options, providers=['CUDAExecutionProvider'])

# input: (b, 3, 512, 512) 
# output: (b, 3, 512, 512) && (b, 1, 512, 512)

n = 1
input_img = np.ones((n, 3, 512, 512), dtype=np.float32)
output_img = np.ones((n, 8, 510, 510), dtype=np.float32)

# X is numpy array on cpu
X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input_img, 'cuda', 0)
Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(output_img, 'cuda', 0)
io_binding = session.io_binding()
io_binding.bind_input(name="X", device_type=X_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
io_binding.bind_output(name="Y", device_type=Y_ortvalue.device_name(), device_id=0, element_type=np.float32, shape=Y_ortvalue.shape(), buffer_ptr=Y_ortvalue.data_ptr())

# Y = io_binding.copy_outputs_to_cpu()[0]

all_time = 0.0
count = 0
loop_count = 100
for i in range(loop_count):
  t0 = time.time() * 1000.0
  session.run_with_iobinding(io_binding)
  t1 = time.time() * 1000.0
  cur_cost = t1 - t0
  print(cur_cost)
  if i > 10:
    all_time += cur_cost
    count += 1

if loop_count > 10:
  print("average time: ", all_time / count)
