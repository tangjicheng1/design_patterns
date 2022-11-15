# 统计某个node的个数

import sys
import onnx 

def count_node(model):
  graph = model.graph
  nodes = graph.node
  nodes_count = {}
  for i in nodes:
    if str(i.op_type) in nodes_count:
      nodes_count[str(i.op_type)] += 1
    else:
      nodes_count[str(i.op_type)] = 1
  
  for key in nodes_count:
    print(key, ":", nodes_count[key])
  return nodes_count
    

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("usage: python count_node.py model.onnx")
    sys.exit(1)
  model = onnx.load(sys.argv[1])
  nodes_count = count_node(model)
