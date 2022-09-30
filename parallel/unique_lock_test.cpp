#include <iostream>
#include <thread>
#include <mutex>

void test3() {
  std::mutex m1;
  m1.lock();
  m1.unlock();
  std::unique_lock<std::mutex> um(m1);
  // um.lock(); // already locked, abort
  um.unlock();
  // um.unlock(); // abort
}

struct A {
  float val;
  void print();
  void lock() {}
  void unlock() {}
};

void test2() {
  std::unique_lock<A> u_a1;
}

void test1() {
  std::mutex m1;
  std::unique_lock<std::mutex> um1(m1);
  auto um2 = std::move(um1);
  return;
}

int main() {
  // test1();
  // test2();
  test3();
  return 0;
}