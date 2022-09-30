#include <iostream>
#define LEN 10

int len_foo() {
  int i = 2;
  return i;
}

constexpr int len_foo_constexpr() { return 5; }
constexpr int fibonacci(const int n) { return n == 1 || n == 2 ? 1 : fibonacci(n - 1) + fibonacci(n - 2); }

int main() {
  char arr_1[10];   // 合法
  char arr_2[LEN];  // 合法
  int len = 10;
  // char arr_3[len];                  // 非法
  const int len_2 = len + 1;
  constexpr int len_2_constexpr = 1 + 2 + 3;
  // char arr_4[len_2];                // 非法
  char arr_4[len_2_constexpr];  // 合法
  // char arr_5[len_foo()+5];          // 非法
  char arr_6[len_foo_constexpr() + 1];  // 合法
  std::cout << fibonacci(10) << std::endl;
  // 1, 1, 2, 3, 5, 8, 13, 21, 34, 55
  std::cout << fibonacci(10) << std::endl;
  return 0;
}