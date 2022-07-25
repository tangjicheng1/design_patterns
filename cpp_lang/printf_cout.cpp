#include <iostream> 
#include <cstdio>
#include <ctime>
#include <chrono>
#include <string>
#include <thread>
#include <ratio>
#include <string_view>

inline double now_time() {
  auto t1 = std::chrono::steady_clock::now();

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  auto t2 = std::chrono::steady_clock::now();

  auto d = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t2 - t1);

  std::cout << d.count() << std::endl;

  return 0.0;
}

int print_times = 1000;

int print_time() {
  std::string print_string("Hello World! Hello World! Hello World! Hello World! Hello World! Hello World! Hello World! ");

  auto t1 = std::chrono::steady_clock::now();

  const char* s = print_string.c_str();

  for (int i = 0; i < print_times; ++i) {
    printf("%s\n", s);
  }

  auto t2 = std::chrono::steady_clock::now();

  auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  // std::cout << "printf: " << cost_time.count() << " ms" << std::endl;
  return cost_time.count();
}

int cout_time() {
  std::string print_string("Hello World! Hello World! Hello World! Hello World! Hello World! Hello World! Hello World! ");

  auto t1 = std::chrono::steady_clock::now();

  for (int i = 0; i < print_times; ++i) {
    std::cout << print_string << std::endl;
  }

  auto t2 = std::chrono::steady_clock::now();

  auto cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
  // std::cout << "cout: " << cost_time.count() << " ms" << std::endl;
  return cost_time.count();
}

int main() {
  int print_cost = print_time();
  int cout_cost = cout_time();
  std::cout << "printf: " << print_cost << " ms" << std::endl;
  std::cout << "cout: " << cout_cost << " ms" << std::endl;

  return 0;
}