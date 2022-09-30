#include <iostream>

class A {
 public:
  A(const std::string& str) {
    std::cout << "Call A(const std::string& str)" << std::endl;
    std::cout << "str == " << str << std::endl;
  }
  A() = default;  // 如果没有这句，会编译出错，因为B的默认构造函数会调用A的默认构造函数
  ~A() = default;
};

class B {
 public:
  B() = default;
  ~B() = default;

 private:
  A a_;
};

int main() {
  B b;
  std::cout << "OK." << std::endl;
  return 0;
}