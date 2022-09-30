#include <iostream>
#include <type_traits>

class A {
  int x;
  double y;
  char* str;
};

struct B {
  double a;
  int b;
  int* c;
};

struct C {
  B a;
  int b;
  double x;
};

void test() {
  std::cout << std::boolalpha << std::is_trivial<int>::value << std::endl;
  std::cout << std::boolalpha << std::is_trivial<A>::value << std::endl;
  std::cout << std::boolalpha << std::is_trivial<B>::value << std::endl;
  std::cout << std::boolalpha << std::is_trivial<C>::value << std::endl;
}

int main() {
  test();
  return 0;
}