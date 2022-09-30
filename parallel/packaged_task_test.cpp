#include <cstdio>
#include <functional>
#include <future>
#include <iostream>
#include <thread>

void test1() {
  std::packaged_task<int(int, int)> task1([](int x, int y) { return (x + y); });
  std::future<int> f1 = task1.get_future();
  task1(1, 2);
  std::cout << f1.get() << std::endl;
}

void worker(std::packaged_task<int(int, int)>& task) { task(1, 2); }

void test2() {
  std::packaged_task<int(int, int)> task2([](int x, int y) { return (x + y); });
  std::future<int> f2 = task2.get_future();
  std::thread t1(worker, std::ref(task2));
  std::cout << f2.get() << std::endl;
  t1.join();
  return;
}

int main() {
  // test1();
  test2();
  return 0;
}