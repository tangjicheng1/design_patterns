#include <array>
#include <iostream>

void test1() {
  std::array<int, 3> a = {1, 2, 3};
  std::array<int, 3> b = {4, 5, 6};
  a.swap(b);

  std::cout << "a.front() = " << a.front() << std::endl;
  std::cout << "a.back() = " << a.back() << std::endl;
  std::cout << "a.data() = " << a.data() << std::endl;

  for (auto iter : a) {
    std::cout << iter << std::endl;
  }

  a.fill(-1);
  for (auto i : a) {
    std::cout << i << std::endl;
  }

  std::cout << std::get<0>(a) << std::endl;
  // std::cout << std::get<3>(a) << std::endl;
  std::cout << a.size() << std::endl;
}

int main() {
  test1();
  return 0;
}