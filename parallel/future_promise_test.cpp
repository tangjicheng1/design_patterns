#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <future>
#include <iostream>
#include <thread>
#include <chrono>

int gen_num() {
  printf("Calling gen_num()...\n");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  return rand() % 100;
}

void worker(std::promise<int>& p) { p.set_value(gen_num()); }

void test1() {
  std::promise<int> prom;
  std::future<int> f;
  std::cout << f.valid() << std::endl;
  f = prom.get_future();
  std::cout << f.valid() << std::endl;
  std::thread t(worker, std::ref(prom));
  // f.wait();
  std::cout << "f.get(): " << f.get() << std::endl;
  t.join();
}

int main() {
  srand((unsigned)time(NULL));
  test1();
  return 0;
}