#include <iostream>

// #define POINTER_TO_BASE_OBJ

struct A {
  virtual void print() { std::cout << "A\n"; }
  int x = 1;
};

struct B : A {
  virtual void print() { std::cout << "B\n"; }
  double y = 2.0;
};

int main() {
#ifdef POINTER_TO_BASE_OBJ
  A* a = new A();
#else
  A* a = new B();
#endif

  B* b = nullptr;
  b = dynamic_cast<B*>(a);
  if (b == nullptr) {
    std::cout << "dynamic b == nullptr\n";
  }
  b = static_cast<B*>(a);
  if (b == nullptr) {
    std::cout << "static b == nullptr\n";
  }
  std::cout << b->y << std::endl;

  delete a;

  return 0;
}