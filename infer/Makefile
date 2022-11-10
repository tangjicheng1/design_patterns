all: ort_test trt_test

ort_test: ort_test.cpp
	g++ -std=c++11 -g -Wall -O3 ort_test.cpp -I /usr/local/cuda/include -L /usr/local/cuda/lib64 \
	-lonnxruntime -lonnxruntime_providers_cuda -lcudart -o ort_test 

trt_test: trt_test.cpp 
	g++ -std=c++11 -g -Wall -O3 trt_test.cpp -I /usr/local/cuda/include -I /opt/TensorRT-8.4.3.1/include \
	-L /usr/local/cuda/lib64 -L /opt/TensorRT-8.4.3.1/lib -lcudart -lnvinfer -lnvonnxparser -o trt_test

.PHONY: clean
clean:
	-rm -rf ort_test
	-rm -rf trt_test