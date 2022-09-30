#include <cstdio>
#include <string>
#include <thread>

void print(int i, std::string str) {
  printf("%d : %s\n", i, str.c_str());
  return;
}

std::thread func1(int i, const std::string& str) {
  printf("Calling func1...\n");
  return std::thread(print, i, str);
}

void func2(std::thread t) {
  printf("Calling func2...\n");
  t.join();
  t = std::thread(print, 1, "hello");
  t.join();
  return;
}

void test1() {
  std::thread t1 = func1(0, "nihao");
  t1.join();
  return;
}

void test2() {
  std::thread t1 = std::thread(print, -1, "in_test2");
  func2(std::move(t1));
  printf("t1.joinable: %d\n", t1.joinable());
  return;
}

int main() {
  // test1();
  test2();
  return 0;
}