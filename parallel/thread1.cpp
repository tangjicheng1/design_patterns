#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

void test1(){
  std::cout << "logic hardware thread count: " << std::thread::hardware_concurrency() << std::endl;
}

int main() {
  test1();
  return 0;
}