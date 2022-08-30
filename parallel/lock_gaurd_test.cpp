#include <cstdio>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

std::mutex m_print;
double global_val = 0.0;


void print(int index, std::string str) {
  std::lock_guard<std::mutex> lock_gaurd_for_print(m_print);
  printf("%d, %s\n", index, str.c_str());
  global_val = global_val + 1.0;
  return; 
}

void test1() {
  std::vector<std::thread> thread_vec;
  int thread_num = 32;
  for (int i = 0; i < thread_num; i++) {
    thread_vec.push_back(std::thread(print, i, std::to_string(i)));
  }

  for (auto& iter : thread_vec) {
    iter.join();
  }

  printf("result: %lf\n", global_val);

  return;
}

int main() {
  test1();
  return 0;
}