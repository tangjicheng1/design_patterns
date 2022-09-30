// builder模式

#include <iostream>
#include <memory>
#include <string>

class Builder {
 public:
  void DoSomething() {
    BuildPartA();
    BuildPartB();
    BuildPartC();
  }

 private:
  virtual void BuildPartA() { std::cout << "Builder::BuildPartA" << std::endl; }
  virtual void BuildPartB() { std::cout << "Builder::BuildPartB" << std::endl; }
  virtual void BuildPartC() { std::cout << "Builder::BuildPartC" << std::endl; }
};

class ChinaBuilder : public Builder {
 private:
  void BuildPartA() override { std::cout << "ChinaBuilder::BuildPartA, China No.1" << std::endl; }
  void BuildPartB() override { std::cout << "ChinaBuilder::BuildPartB, China No.2" << std::endl; }
  void BuildPartC() override { std::cout << "ChinaBuilder::BuildPartC, China No.3" << std::endl; }
};

class JapanBuilder : public Builder {
 private:
  void BuildPartA() override { std::cout << "JapanBuilder::BuildPartA, Japan No.1" << std::endl; }
  void BuildPartB() override { std::cout << "JapanBuilder::BuildPartB, Japan No.2" << std::endl; }
  void BuildPartC() override { std::cout << "JapanBuilder::BuildPartC, Japan No.3" << std::endl; }
};

void test1() {
  std::unique_ptr<Builder> cn_builder(new ChinaBuilder());
  cn_builder->DoSomething();

  std::unique_ptr<Builder> jp_builder(new JapanBuilder());
  jp_builder->DoSomething();
  return;
}

class Test2 {
 public:
  Test2() : cn_builder_(new ChinaBuilder()), jp_builder_(new JapanBuilder()) {}
  Test2(int val) {
    if (val == 0) {
      cn_builder_.reset(nullptr);
      jp_builder_.reset(nullptr);
    } else {
      cn_builder_.reset(new ChinaBuilder());
      jp_builder_.reset(new JapanBuilder());
    }
  }

  void test2() {
    if (cn_builder_ != nullptr) {
      cn_builder_->DoSomething();
    } 
    if (jp_builder_ != nullptr) {
      jp_builder_->DoSomething();
    }
  }

 private:
  std::unique_ptr<Builder> cn_builder_;
  std::unique_ptr<Builder> jp_builder_;
};

int main() {
  // test1();
  Test2 t2;
  t2.test2();
  Test2 t3(1);
  t3.test2();
  return 0;
}