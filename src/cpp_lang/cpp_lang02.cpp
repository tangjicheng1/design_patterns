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

void test02(){}

int main() {
  test01();

  return 0;
}