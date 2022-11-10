ort_test: ort_test.cpp
	g++ -std=c++11 -g -Wall -O3 -I /usr/local/cuda/include ort_test.cpp -L /usr/local/cuda/lib64 \
	-lonnxruntime -lonnxruntime_providers_cuda -lcudart -o ort_test 

.PHONY: clean
clean:
	-rm -rf ort_test