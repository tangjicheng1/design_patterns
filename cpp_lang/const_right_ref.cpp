#include <cstdio>
#include <functional>
#include <string>
#include <vector>

struct B {
  B(int i) : val(i) {}

  int val;
};

// A只接受左值引用，不接受右值引用
struct A {
  A(const B& b) : val(b.val) {}

  A(const B&& b) = delete;

  int val;
};

void test1() {
  int i = 1;
  int j = 2;
  auto r1 = std::ref(i);
  // auto r2 = std::ref(i + j);

  B b1(1);
  B b2(2);
  A a1(b1);
  // A a2(B(3));  // 编译失败，因为A接受了右值，而右值引用的构造函数是delete的
}

int main() {
  test1();
  return 0;
}