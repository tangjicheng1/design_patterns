#include <iostream>
#include <type_traits>
#include <vector>

template <typename T>
int len(T t) {
  if constexpr (std::is_integral<T>::value) {
    return t;
  } else {
    return t.size();
  }
}

int length(int i) { return len(i); }

int length(std::vector<int> i) { return len(i); }

void test() {
  int i = 100;
  std::vector<int> v(10, 0);
  std::cout << length(i) << std::endl;
  std::cout << length(v) << std::endl;
  ;
}

int main() {
  test();
  return 0;
}