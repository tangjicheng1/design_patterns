// bridge模式

#include <iostream>
#include <memory>
#include <string>

class DrawImpl {
 public:
  virtual ~DrawImpl() = default;
  virtual void Draw() = 0;
};

class Shape {
 public:
  virtual ~Shape() = default;
  virtual void Draw() = 0;

 protected:
  std::unique_ptr<DrawImpl> impl_;
};

class Red : public DrawImpl {
 public:
  void Draw() override { std::cout << "This Color is Red" << std::endl; }
};

class Blue : public DrawImpl {
 public:
  void Draw() override { std::cout << "This Color is Blue" << std::endl; }
};

class Circle : public Shape {
 public:
  Circle() { impl_.reset(new Red()); }

  void Draw() override {
    std::cout << "This is Circle \n";
    impl_->Draw();
  }
};

void test1() {
  Circle c;
  c.Draw();
}

int main() {
  test1();
  return 0;
}