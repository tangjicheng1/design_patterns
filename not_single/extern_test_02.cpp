#include "extern_test_01.h"

#include <iostream>

void test02() {
  std::cout << "test 2:\n";
  // std::cout << var_without_extern << std::endl;
  std::cout << "var_with_static: " << var_with_static << std::endl;
  std::cout << "var_with_extern: " << var_with_extern << std::endl;
}

void test01();

int main() {
  test01();
  test02();
  return 0;
}