#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <iostream>

std::atomic_flag f1 = ATOMIC_FLAG_INIT;

void test1() {
  std::cout << std::boolalpha << f1.test_and_set() << std::endl;
  std::cout << std::boolalpha << f1.test_and_set() << std::endl;
  std::future<void> fu1 = std::async(std::launch::async, []() { f1.clear(); });
  std::cout << std::boolalpha << f1.test_and_set() << std::endl;
  std::this_thread::sleep_for(std::chrono::seconds(1));
  std::cout << std::boolalpha << f1.test_and_set() << std::endl;
}

int main() {
  test1();
  return 0;
}