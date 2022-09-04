#include <iostream>
#include <string>
#include <vector>

void if_test() {
  std::vector<int> vec{1, 2, 3, 4};
  if (auto iter = vec.begin(); *iter == 1) {
    std::cout << "vec first is 1" << std::endl;
  }
}

void switch_test() {
  std::vector<int> vec{1, 2, 3};
  std::vector<int> vec2{2, 3};
  switch (auto iter = vec.begin(); *iter) {
    case 1:
      std::cout << "vec first is 1" << std::endl;
      break;
    case 2:
      std::cout << "vec first is 2" << std::endl;
      break;
    default:
      std::cout << "vec first is other" << std::endl;
      break;
  }
}

int main() {
  std::cout << "if test\n";
  if_test();
  std::cout << "switch test\n";
  switch_test();
  return 0;
}