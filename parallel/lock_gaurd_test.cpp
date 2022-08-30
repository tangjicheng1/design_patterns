#include <cstdio>
#include <mutex>
#include <thread>
#include <string>
#include <vector>
#include <chrono>

std::mutex m_global;
double global_val = 0.0;


void print(int index, std::string str) {
  // std::lock_guard<std::mutex> lock_gaurd_for_global_val(m_global);
  printf("%d, %s\n", index, str.c_str());
  std::chrono::milliseconds dur(10);

  double temp = global_val;
  std::this_thread::sleep_for(dur);
  global_val = temp + 1.0;

  return; 
}

void test1() {
  std::vector<std::thread> thread_vec;
  int thread_num = 10;
  for (int i = 0; i < thread_num; i++) {
    thread_vec.push_back(std::thread(print, i, std::to_string(i) + "_thread"));
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