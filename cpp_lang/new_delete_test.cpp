#include <cstdio>
#include <string>
#include <algorithm>
#include <functional>

struct A final {
  int num;
  double val;
  void print() { printf("Calling A::print()\n"); }
};

void test1() {
  int* p = new int(7);
  printf("[1] %d\n", *p);
  delete p;

  int* q = new int[5];
  printf("[2] %s\n", "test");
  delete[] q;

  int n = 3;
  A* p2 = new A[n];

  std::for_each(p2, p2 + n, std::mem_fn(&A::print));

  delete[] p2;

  return;
}

int main() {
  test1();
  return 0;
}