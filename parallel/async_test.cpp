#include <chrono>
#include <cstdio>
#include <functional>
#include <future>
#include <iostream>

int id = 0;

int simple_worker() {
  std::this_thread::sleep_for(std::chrono::seconds(3));
  printf("Calling simple 1\n");
  return id;
}

int simple_worker2() {
  printf("Calling simple 2 \n");
  return id;
}

void test1() {
  std::async(std::launch::async, simple_worker);
  std::async(std::launch::async, simple_worker2);
}

int main() {
  test1();
  return 0;
}