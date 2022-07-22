// 外观模式，为同一类子系统提供统一的对外接口
#include <iostream>

class Sub {
 public:
  virtual void func1() = 0;
  virtual void func2() = 0;
};

class Sub1 : public Sub {
 public:
  void func1() override { std::cout << "Call Sub1 func1." << std::endl; }
  void func2() override { std::cout << "Call Sub1 func2." << std::endl; }
};

class Sub2 : public Sub {
 public:
  void func1() override { std::cout << "Call Sub2 func1." << std::endl; }
  void func2() override { std::cout << "Call Sub2 func2." << std::endl; }
};

class Client {
 public:
  Client() {
    this->sub1_ = new Sub1();
    this->sub2_ = new Sub2();
  }
  void start() {
    sub1_->func1();
    sub2_->func1();
    sub1_->func2();
    sub2_->func2();
  }

 private:
  Sub1* sub1_;
  Sub2* sub2_;
};

int main() {
  Client client;
  client.start();
  return 0;
}