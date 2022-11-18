INCLUDE_PATH := -I ./include 
CC_FLAG := -std=c++11 -g -gencode=arch=compute_75,code=sm_75
CC := nvcc

all : test1 test2

test1 : test1.cu
	$(CC) ${CC_FLAG} ${INCLUDE_PATH} -lcudart test1.cu -o test1

test2 : test2.cu
	$(CC) ${CC_FLAG} ${INCLUDE_PATH} -lcudart test2.cu -o test2


.PHONY: clean
clean: 
	-rm -rf test1 test2