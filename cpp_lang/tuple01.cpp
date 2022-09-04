#include <iostream>
#include <string>
#include <tuple>

void test02() {
  int a = 1;
  double b = 1.2;
  auto tuple1 = std::make_tuple(a, b);
  std::cout << std::get<0>(tuple1) << std::endl;
}

void test01() {
  std::tuple<int, int> a{1, 2};
  std::cout << std::get<0>(a) << " , " << std::get<1>(a) << std::endl;
  std::tuple<int, int> b{3, 4};
  std::cout << std::get<0>(b) << " , " << std::get<1>(b) << std::endl;

  std::swap(a, b);
  std::cout << std::get<0>(a) << " , " << std::get<1>(a) << std::endl;
  std::cout << std::get<0>(b) << " , " << std::get<1>(b) << std::endl;

  a.swap(b);
  std::cout << std::get<0>(a) << " , " << std::get<1>(a) << std::endl;
  std::cout << std::get<0>(b) << " , " << std::get<1>(b) << std::endl;

  b.swap(a);
  std::cout << std::get<0>(a) << " , " << std::get<1>(a) << std::endl;
  std::cout << std::get<0>(b) << " , " << std::get<1>(b) << std::endl;
}

int main() {
  test02();
  // test01();
  return 0;
}