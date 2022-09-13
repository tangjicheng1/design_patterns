#include <iostream>
#include <array>
#include <vector>
#include <cstdio>
#include <span>

void test01() {
  std::vector<float> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::span<float> span1(vec.data(), vec.size());
  for (auto iter : span1) {
    std::cout << iter << std::endl;
  }
}

int main() {
  test01();
  return 0;
}