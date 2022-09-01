#include <thread>

void test1() {
  std::thread t1([]() { return 0; });
  t1.detach();
}

int main() {
  test1();
  return 0;
}