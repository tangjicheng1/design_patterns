#include <iostream>

void foo(int i) {
  std::cout << "int:" << i << std::endl;
}

void foo(char* str) {
  if (str == nullptr) {
    std::cout << "str: nullptr\n";
  }
  std::cout << "str:" << str << std::endl;
}

void test() {
  // foo(NULL);
  foo(0);
  foo(nullptr);
}

int main() {
  test();
  return 0;
}