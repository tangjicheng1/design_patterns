CC := g++
CC_FLAGS := -std=c++11 -g -Wall -O3

LD_FLAGS := -L /usr/local/cuda/lib64 -L /opt/TensorRT-8.4.3.1/lib 
INCLUDE_FLAGS := -I /usr/local/cuda/include -I /opt/TensorRT-8.4.3.1/include

all: residual_block_test

residual_block_test: residual_block_test.cpp 
	$(CC) $(CC_FLAGS) $(LD_FLAGS) $(INCLUDE_FLAGS) residual_block_test.cpp \
	residual_block.cpp \
	 -lcudart -lcudnn -o residual_block_test

.PHONY: clean
clean:
	-rm -rf residual_block_test