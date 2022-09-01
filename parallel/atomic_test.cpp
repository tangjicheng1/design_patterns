#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <string>

struct A {
  int num;
  double val;
};

int worker1(std::atomic_int& input) {
  for (int i = 0; i < 10; i++) {
    input.store(i);
  }
  return input.load();
}

void test1() {
  std::cout << "test1:" << std::endl;
  std::atomic_int x;
  std::future<int> fu1 = std::async(std::launch::async, worker1, std::ref(x));
  std::cout << std::boolalpha << x.is_lock_free() << std::endl;
  std::cout << fu1.get() << std::endl;
}

A worker2(std::atomic<A>& input) {
  for (int i = 0; i < 10; i++) {
    A a{i, i * 10.0};
    input.store(a);
  }
  return input.load();
}

void test2() {
  std::cout << "test2:" << std::endl;
  std::atomic<A> x;
  std::future<A> fu1 = std::async(std::launch::async, worker2, std::ref(x));
  std::cout << std::boolalpha << x.is_lock_free() << std::endl;
  std::cout << fu1.get().val << std::endl;
}

int main() {
  test1();
  test2();
  return 0;
}