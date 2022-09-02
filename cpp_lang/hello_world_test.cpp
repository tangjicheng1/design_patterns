#include <cstdio>
#include <iostream>

void print_cpp(int x) {
  std::cout << x << std::endl;
}

extern "C" {
  void print(int x) {
    print_cpp(x);
  }
}

int main() {
  printf("Hello, world\n");
  return 0;
}