#include <iostream>

struct A {
  A() {
    ptr = new float[10];
    std::cout << "A()" << std::endl;
  }

  ~A() {
    delete[] ptr;
    std::cout << "~A()" << std::endl;
  }

  float* ptr;
};

int main() {
  A a;
  for (int i = 0; i < 10; i++) {
    a.ptr[i] = i;
  }
  for (int i = 0; i < 10; i++) {
    std::cout << a.ptr[i] << std::endl;
  }
  return 0;
}