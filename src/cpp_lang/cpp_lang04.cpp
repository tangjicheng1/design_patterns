// 析构函数，虚函数
#include <iostream>

class A {
 public:
  A() { std::cout << "Call A()" << std::endl; }
  // ~A()前面应该加virtual
  virtual ~A() { std::cout << "Call ~A()" << std::endl; }
};

class B : public A {
 public:
  B() { std::cout << "Call B()" << std::endl; }
  ~B() { std::cout << "Call ~B()" << std::endl; }
};

int main() {
  A* ptr_b = new B();
  delete ptr_b;
  return 0;
}

/*
不加virtual的输出如下：
Call A()
Call B()
Call ~A()
*/
/*
加virtual的输出如下：
Call A()
Call B()
Call ~B()
Call ~A()
*/