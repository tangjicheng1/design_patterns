#include <future>
#include <iostream>
#include <cstdio>
#include <functional>

int id = 0;

int simple_worker() {
  id += 1;
  printf("Calling simple %d\n", id);
  return id;
}

void test1() {
  auto f1 = std::async(std::launch::async, simple_worker);
  auto f2 = std::async(std::launch::async, simple_worker);
  
  std::cout << f1.get() << std::endl;
  std::cout << f2.get() << std::endl;
}

int main() {
  test1();
  return 0;
}