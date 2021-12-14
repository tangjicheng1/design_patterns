#include <iostream>

int Print() {
  std::cout << "Call Print() " << std::endl;
  return 0;
}

class A {
 public:
  A() { std::cout << "Call A() " << count++ << std::endl; }
  static int count;
};
int A::count = 0;

// 通过定义静态对象，调用静态对象的构造函数
// 这个构造函数会在进入main函数之前调用
A* static_ptr_a = new A();
A a;

// 通过调用函数给全局静态变量赋值。
// Print()也会在main()之前调用
int static_i = Print();

// Print(); // 会有编译错误

int main() {
  std::cout << "Call main() " << std::endl;
  return 0;
}