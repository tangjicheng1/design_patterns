//
// Created by 汤继承 on 2022/8/6.
//
#include "array_ref.h"
#include <iostream>

void test1() {
  int arr_c_style[3] = {1, 2, 3};
  ArrayRef<int> ref1(arr_c_style, 3);
  for (auto iter : ref1) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
  // ref1[1] = -1; // error: 只能const访问
}

int main() {
  test1();
  return 0;
}