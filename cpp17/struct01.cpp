#include <iostream>
#include <string>
#include <vector>

struct A {
  int a;
  double b;
  std::string s;
  void print() {
    std::cout << a << " , " << b << " , " << s << std::endl;
  }
};

void test01() {
  A a_struct{1, 2.0, "hello"};
  // []内必须是全部的struct的成员变量
  auto& [x, y, str] = a_struct;
  x = 4;
  a_struct.print();
}

int main() {
  test01();
  return 0;
}