#include <iostream> 
#include <cstdio>
#include <ctime>
#include <chrono>
#include <string>
#include <thread>
#include <ratio>
#include <string_view>

// #include <spdlog/spdlog.h>

const char* sen = "Hello world, this is a test, OK, one two three";
int N = 1000;
int test_count = 10;

void printf_test() {
  for (int i = 0; i < N; i++) {
    printf("%s\n", sen);
  }
}

void cout_test() {
  for (int i = 0; i < N; i++) {
    std::cout << sen << std::endl;
  }
}

// void spdlog_test() {
//   for (int i = 0; i < N; i++) {
//     spdlog::info(sen);
//   }
// }

void test() {
  for (int i = 0; i < test_count; i++) {
    auto t0 = std::chrono::steady_clock::now();
    printf_test();
    auto t1 = std::chrono::steady_clock::now();
    cout_test();
    auto t2 = std::chrono::steady_clock::now();
    // spdlog_test();
    auto t3 = std::chrono::steady_clock::now();

    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
    std::cout << "printf: " << duration1.count() << " us" << std::endl;
    std::cout << "cout: " << duration2.count() << " us" << std::endl;
    std::cout << "spdlog: " << duration3.count() << " us" << std::endl;
  }
}

int main() {
  test();

  return 0;
}