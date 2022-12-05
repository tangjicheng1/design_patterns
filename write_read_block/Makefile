all: test cv_test shared

test : test.cpp
	g++ -std=c++11 test.cpp -o test 

cv_test: cv_test.cpp
	g++ -std=c++11 cv_test.cpp -lpthread -o cv_test

shared: shared.cpp 
	g++ -std=c++17 shared.cpp -lpthread -o shared

.PHONY: clean
clean:
	-rm -rf test cv_test shared