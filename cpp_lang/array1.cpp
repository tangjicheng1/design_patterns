//
// Created by 汤继承 on 2022/8/6.
//
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

void test1() {
  std::array<double, 3> a{1, 2, 3};
  std::cout << "First loop" << std::endl;
  std::cout << a.front() << " " << a.at(1) << " " << a.back() << std::endl;
  std::cout << "Second loop" << std::endl;
  for (auto iter : a) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
  std::cout << "Third loop" << std::endl;
  std::for_each(a.begin(), a.end(), [](double x) { std::cout << x << " "; });
  std::cout << std::endl;
}

void test2() {
  using int_pair = std::array<int, 2>;
  std::array<int_pair, 3> pair_3;
  pair_3.fill({1, 2});
  for (auto iter : pair_3) {
    for (auto iter2 : iter) {
      std::cout << iter2 << " ";
    }
    std::cout << std::endl;
  }
}

template <typename T, size_t N>
void print_array(const std::array<T, N>& arr) {
  for (auto iter : arr) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
}

void test3() {
  std::array<int, 2> arr1{1,2};
  std::array<int, 2> arr2{3, 4};
  arr1[0] = 100;
  std::swap(arr1, arr2);
  std::cout << "arr1:" << std::endl;
  print_array(arr1);
  std::cout << "arr2:" << std::endl;
  print_array(arr2);
}

int main() {
  // test1();
  // test2();
  test3();
  return 0;
}