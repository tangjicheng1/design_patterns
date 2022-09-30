#include <iostream>
#include <typeinfo>
#include <vector>
#include <string>

struct A {
  int val;
  void print() {
    std::cout << "A::print" << std::endl;
  }
};

void test1() {
  std::vector<int> v1;
  std::vector<std::string> vs;
  int* p1;
  double* p2;
  std::vector<double*> vp;
  A a;
  std::vector<A> va;

  std::cout << "std::vector<int>: " << typeid(v1).name() << std::endl;
  std::cout << "std::vector<std::string>: " << typeid(vs).name() << std::endl;
  std::cout << "int* p1: " << typeid(p1).name() << std::endl;
  std::cout << "double* p2: " << typeid(p2).name() << std::endl;
  std::cout << "std::vector<double*>: " << typeid(vp).name() << std::endl;
  std::cout << "A: " << typeid(a).name() << std::endl;
  std::cout << "std::vector<A>: " << typeid(va).name() << std::endl;
}

int main() {
  test1();
  return 0;
}