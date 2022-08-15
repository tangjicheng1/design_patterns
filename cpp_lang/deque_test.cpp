#include <deque>
#include <iostream>

void test1() {
  std::deque<int> a{1, 2, 3, 4, 5};
  for (auto i : a) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  std::deque<int> b;
  b.assign(a.begin(), a.end());
  std::cout << "b.size() = " << b.size() << std::endl;
  for (auto i : b) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  std::cout << b.front() << " " << b.back() << std::endl;
  b.push_front(-1);
  b.push_back(6);
  for (auto i : b) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  b.emplace(b.begin() + 1, -2);
  b.pop_front();
  for (auto i : b) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
}

int main() {
  test1();
  return 0;
}