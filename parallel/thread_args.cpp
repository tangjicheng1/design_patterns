#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

void print(int i, const std::string& str) {
  printf("[print] %d : %s\n", i, str.c_str());
  return;
}

void set_str(std::string& str) {
  str.append("_Add_Hello");
  return;
}

void test1() {
  std::vector<std::thread> threads_vec;
  int N = 10;
  for (int i = 0; i < N; i++) {
    // 线程参数使用值传递
    threads_vec.push_back(std::thread(print, i, std::to_string(i) + "_str"));
  }

  for (int i = 0; i < N; i++) {
    threads_vec[i].join();
  }

  return;
}

void test2() {
  std::vector<std::string> str_vec(10, "");
  for (int i = 0; i < 10; i++) {
    str_vec[i] = "origin_" + std::to_string(i) + "_";
  }

  std::vector<std::thread> threads_vec;
  for (int i = 0; i < 10; i++) {
    threads_vec.push_back(std::thread(set_str, std::ref(str_vec[i])));
  }
  for (int i = 0; i < 10; i++) {
    threads_vec[i].join();
  }

  for (int i = 0; i < 10; i++) {
    printf("%d: %s\n", i, str_vec[i].c_str());
  }
  return;
}

int main() {
  // test1();
  test2();
  return 0;
}