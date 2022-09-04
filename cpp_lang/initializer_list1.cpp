#include <initializer_list>
#include <iostream>

struct A {
  A() { std::cout << "A()" << std::endl; }
  A(const A&) { std::cout << "A(const A&)" << std::endl; }
  ~A() { std::cout << "~A()" << std::endl; }
};

void funcA(A a) { std::cout << "funcA(A a)" << std::endl; }

void funcRefA(const A& a) { std::cout << "funcRefA(const A& a)" << std::endl; }

void func(const std::initializer_list<int>& il) {
  for (auto& i : il) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
}

void test1() {
  std::initializer_list<int> il = {1, 2, 3};
  func(il);
  func({4, 5, 6});
}

void test2() {
  A a;
  funcA(a);
}

void test3() {
  A a;
  funcRefA(a);
}

int main() {
  // test1();
  // test2();
  test3();
  return 0;
}