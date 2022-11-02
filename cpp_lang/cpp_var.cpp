#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

// 实现C++可变参数的方法

// 第一种，如果参数的类型相同，可以考虑使用initializer_list，
// C++11新特性，但实际上，是输入了一个参数，而不是多个，需要加入大括号。

template <typename T>
void print1(const std::initializer_list<T>& input) {
  for (const auto& iter : input) {
    std::cout << iter << " ";
  }
  std::cout << std::endl;
}

void test1() {
  print1<int>({});
  print1<double>({1, 2, 3, 4.5});
  print1({1, 2, 3, 4, 5});
}

// 第二种，使用C++ template的特性
// 但是，比较麻烦的一点是，需要递归调用，去解开Args...
// 而且，需要实现两个解包函数，一个适用于终止条件，另一个适用于递归条件
template <typename T>
void unpack_print2(T last) {
  std::cout << last << " ";
  std::cout << std::endl;
}

template <typename T, typename... Args>
void unpack_print2(T first, Args... other_input) {
  std::cout << first << " ";
  unpack_print2(other_input...);
}

template <typename... Args>
void print2(Args... input) {
  unpack_print2(input...);
}

void test2() {
  print2(1, 1.4, "Hello");
  print2("Good", std::string("job"), 12.345);
}

// 第三种，是C语言的方式
void print3(const char* str, ...) {
  va_list vp;
  va_start(vp, str);
  if (strcmp(str, "1") == 0) {
    std::cout << "input: ";
    std::cout << va_arg(vp, int) << std::endl;
    va_end(vp);
    return;
  }
  if (strcmp(str, "2") == 0) {
    std::cout << "input: ";
    std::cout << va_arg(vp, int) << " ";
    std::cout << va_arg(vp, int) << std::endl;
    va_end(vp);
    return;
  }
  printf("input error\n");
}
void test3() {
  print3("2", 3, 5);
  print3("1", 3);
  print3("str", 1, 2, 3, 4, 5);
}

int main() {
  test1();
  test2();
  test3();
  return 0;
}