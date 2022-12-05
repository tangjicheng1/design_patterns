#include <iostream>
#include <stdio.h>
#include <thread>
#include <vector>

#include "rw_lock.hpp"

RWLock global_rw_lock;

int buff[1024];

void write1() {
  global_rw_lock.write_lock();
  printf("Write1: ");
  for (int i = 0; i < 1024; i++) {
    buff[i] = 9;
  }
  for (int i = 0; i < 10; i++) {
    printf("%d ", buff[i]);
  }
  printf("\n");
  global_rw_lock.write_unlock();
}

void write2() {
  global_rw_lock.write_lock();
  printf("Write2: ");
  for (int i = 0; i < 1024; i++) {
    buff[i] = 1;
  }
  for (int i = 0; i < 10; i++) {
    printf("%d ", buff[i]);
  }
  printf("\n");
  global_rw_lock.write_unlock();
}

void read() {
  global_rw_lock.read_lock();
  printf("Read: ");
  for (int i = 0; i < 10; i++) {
    printf("%d ", buff[i]);
  }
  printf("\n");
  global_rw_lock.read_unlock();
}

void test() {
  std::vector<std::thread> vec(100);
  for (size_t i = 0; i < vec.size(); i++) {
    if (i % 5 == 0) {
      vec[i] = std::thread(write1);
    }
    else if (i % 5 == 3) {
      vec[i] = std::thread(write2);
    }
    else {
      vec[i] = std::thread(read);
    } 
  }

  for (size_t i = 0; i < vec.size(); i++) {
    vec[i].join();
  }
}

int main() {
  test();
  return 0;
}