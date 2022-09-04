#include "inline01.h"

#include <iostream>

void test_inline01();

void test_inline02() {
  std::cout << "test_inline02" << std::endl;
  std::cout << "a = " << a << std::endl;
}

int main() {
  test_inline01();
  test_inline02();
  return 0;
}