// Pointer to implementation
// Pimpl设计模式：在对外暴露的接口中，隐藏函数的具体实现
#include <iostream>

// .h文件
// 一般的对外接口设计，如果包含private的具体实现，如下
class Widget {
 public:
  void func();

 private:
  void my_func1();  // 对外暴露了私有函数接口
  void my_func2();

 private:
  int val;  // 对外暴露了私有变量
};

// .cpp文件
void Widget::func() {
  my_func1();
  my_func2();
  std::cout << val << std::endl;
  return;
}

void Widget::my_func1() {
  val = 1;
  return;
}

void Widget::my_func2() {
  val += 20;
  return;
}

// 另外一种更好的实现方式是，
// 通过把上述私有函数和变量封装成新的类，不去暴露具体实现细节
// .h文件如下
class Widget2 {
 public:
  void func();

 private:
  class Impl;
  Impl* ptr_impl_;
};

// .cpp文件
class Widget2::Impl{
  public:
  void my_func1();
  void my_func2();
  int get_val();
  private:
  int val_;
};

void Widget2::Impl::my_func1(){
  this->val_ = 1;
  return;
}

void Widget2::Impl::my_func2(){
  this->val_ += 20;
  return;
}

int Widget2::Impl::get_val(){
  return this->val_;
}

void Widget2::func(){
  ptr_impl_ = new Impl();
  ptr_impl_->my_func1();
  ptr_impl_->my_func2();
  std::cout << ptr_impl_->get_val() << std::endl;
  return;
}

void test01(){
  std::cout << "Test 1 : " << std::endl;
  Widget w;
  w.func();
  return;
}

void test02(){
  std::cout << "Test 2 : " << std::endl;
  Widget2 w;
  w.func();
  return;
}

int main(){
  test01();
  test02();
  return 0;
}