all: ort_test trt_test ort_model_opt

CC := g++
CC_FLAGS := -std=c++11 -g -Wall -O3

# onnxruntime library and header has been in system path
# if not, add onnxruntime library path and include path below
LD_FLAGS := -L /usr/local/cuda/lib64 -L /opt/TensorRT-8.4.3.1/lib 
INCLUDE_FLAGS := -I /usr/local/cuda/include -I /opt/TensorRT-8.4.3.1/include

ort_test: ort_test.cpp
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(INCLUDE_FLAGS) ort_test.cpp \
	-lonnxruntime -lonnxruntime_providers_cuda -lcudart -o ort_test 

trt_test: trt_test.cpp 
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(INCLUDE_FLAGS) trt_test.cpp \
	 -lcudart -lnvinfer -lnvonnxparser -o trt_test

ort_model_opt: ort_model_opt.cpp
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(INCLUDE_FLAGS) ort_model_opt.cpp \
	-lonnxruntime -o ort_model_opt 


.PHONY: clean
clean:
	-rm -rf ort_test
	-rm -rf trt_test
	-rm -rf ort_model_opt