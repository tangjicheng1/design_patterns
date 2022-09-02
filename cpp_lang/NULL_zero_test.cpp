#include <iostream>

void foo(int i) {
  std::cout << "int:" << i << std::endl;
}

void foo(char* str) {
  std::cout << "str:" << str << std::endl;
}

void test() {
  // foo(NULL);
  foo(0);
}

int main() {
  test();
  return 0;
}