#include <iostream>

[[nodiscard]] int f(int x) { return x; }

void test() {
  int a = f(1);
  f(2);  // 会有编译警告
}

int main() {
  test();
  return 0;
}