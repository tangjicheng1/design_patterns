#include <array>
#include <cstdio>
#include <iostream>
#include <span>
#include <vector>

void test01() {
  std::vector<float> vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::span<float> span1(vec.data(), vec.size());
  for (auto iter : span1) {
    std::cout << iter << std::endl;
  }
  int a[4] = {1,2,3,4};
  std::span<int> a_span(a);
  std::cout << "size: " << a_span.size() << std::endl;
  std::span<const float> vec_span(vec.data(), vec.size());
  std::cout << "size: " << vec_span.size() << std::endl;
  vec_span[1];
  a_span[2] = 0;
}

int main() {
  test01();
  return 0;
}