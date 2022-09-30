#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <functional>
#include <algorithm>

void free_print(int i) { printf("Call free print: %d\n", i); }

struct A final {
 public:
  void a_print(int i) { printf("Call A print: %d\n", i); }
};

std::once_flag free_print_flag_1;
void call_free_print(int i) {
  std::call_once(free_print_flag_1, free_print, i);
  std::call_once(free_print_flag_1, free_print, i);
}

void test1() {
  std::vector<std::thread> threads;
  int thread_num = 10;
  for (int i = 0; i < thread_num; i++) {
    threads.push_back(std::thread(call_free_print, i));
  }
  std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
  return;
}

void test2() {
  for (int i = 0; i < 10; i++) {
    call_free_print(i);
  }
}

int main() {
  // test1();
  test2();
  return 0;
}