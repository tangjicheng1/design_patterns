conv_test: conv_transpose.cpp conv_transpose_test.cpp
	nvcc -std=c++11 conv_transpose.cpp conv_transpose_test.cpp -lcudnn -o conv_test

.PHONY: clean
clean:
	-rm -rf conv_test