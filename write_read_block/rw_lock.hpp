#pragma once
#include <mutex>
#include <thread>
#include <stdio.h>
class RWLock final {
 public:
  void read_lock() {
    read_count_mutex.lock();
    read_count += 1;
    if (read_count == 1) {
      global_mutex.lock();
    }
    printf("Now read count: %d\n", read_count);
    read_count_mutex.unlock();
  }
  void read_unlock() {
    read_count_mutex.lock();
    read_count -= 1;
    if (read_count == 0) {
      global_mutex.unlock();
    }
    printf("Now read count: %d\n", read_count);
    read_count_mutex.unlock();
  }
  void write_lock() { global_mutex.lock(); }
  void write_unlock() { global_mutex.unlock(); }

 private:
  int read_count = 0;
  std::mutex read_count_mutex;
  std::mutex global_mutex;
};