#include "extern_test_01.h"

#include <iostream>

// 定义
int var_with_extern = 0;

void test01() {
  std::cout << "test 1:\n";
  var_with_static += 100;
  var_with_extern += 100;
  // std::cout << var_without_extern << std::endl;
  std::cout << "var_with_static: " << var_with_static << std::endl;
  std::cout << "var_with_extern: " << var_with_extern << std::endl;
}