// delete基类指针的时候，要注意基类指针应该需要有析构函数

#include <iostream>

class Base {
 public:
  virtual void Print() = 0;
  virtual ~Base() = default;
  // 如果没有这句，编译器有warning
  // -Wdelete-abstract-non-virtual-dtor
};

class FromBase : public Base {
  void Print() override {
    std::cout << "Call FromBase::Print() override." << std::endl;
    return;
  }
};

void test01() {
  Base* ptr = new FromBase();
  ptr->Print();
  delete ptr;
  return;
}

// 测试继承类对基类的构造与析构函数的调用顺序
class A {
 public:
  A() { std::cout << "Call A() " << std::endl; }
  ~A() { std::cout << "Call ~A() " << std::endl; }
};

class B : public A {
 public:
  B() { std::cout << "Call B() " << std::endl; }
  ~B() { std::cout << "Call ~B() " << std::endl; }
};

void test02() {
  B b;
  return;
}

int main() {
  std::cout << "Test 1: " << std::endl;
  test01();
  std::cout << "Test 2 : " << std::endl;
  test02();
  return 0;
}