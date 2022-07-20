#include <cstdio>
#include <iostream>
#include <vector>

namespace adl_test {
namespace a_name {
struct A {
  A(int val) { x = val; }
  int x;
};

void Print(A a) { std::cout << "a_name::print" << a.x << std::endl; }
}  // namespace a_name

namespace b_name {
struct A {
  A(int val) { x = val; }
  int x;
};
void Print(A a) { std::cout << "b_name::print" << a.x << std::endl; }
}  // namespace b_name
}  // namespace adl_test

int main() {
  adl_test::a_name::A a_name_val(1);
  adl_test::b_name::A b_name_val(2);
  Print(a_name_val);
  Print(b_name_val);

  return 0;
}