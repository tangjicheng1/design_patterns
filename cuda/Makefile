conv2d_test: conv_transpose.o conv2d_transpose_test.cpp
	nvcc -std=c++11 conv_transpose.o conv2d_transpose_test.cpp -lcudnn -o conv_test

conv_transpose.o: conv_transpose.cpp 
	nvcc -std=c++11 -c conv_transpose.cpp -o conv_transpose.o

.PHONY: clean
clean:
	-rm -rf conv_test
	-rm -rf *.o