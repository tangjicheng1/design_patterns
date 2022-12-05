#include <shared_mutex>
#include <thread>
#include <vector>

std::shared_mutex global_read_write_mutex;
using ReadLock = std::shared_lock<std::shared_mutex>;
using WriteLock = std::unique_lock<std::shared_mutex>;

int buff[1024];

void write1() {
  WriteLock lock(global_read_write_mutex);
  for (size_t i = 0; i < 1024; i++) {
    buff[i] = 9;
  }
  printf("[Write1] ");
  for (size_t i = 0; i < 10; i++) {
    printf("%d ", buff[i]);
  }
  printf("\n");
}

void write2() {
  WriteLock lock(global_read_write_mutex);
  for (size_t i = 0; i < 1024; i++) {
    buff[i] = 1;
  }
  printf("[Write2] ");
  for (size_t i = 0; i < 10; i++) {
    printf("%d ", buff[i]);
  }
  printf("\n");
}

void read() {
  ReadLock lock(global_read_write_mutex);
  printf("[Read] ");
  for (size_t i = 0; i < 10; i++) {
    printf("%d ", buff[i]);
  }
  printf("\n");
}

int main() {
  std::vector<std::thread> vec(100);
  for (size_t i = 0; i < 100; i++) {
    if (i % 5 == 0) {
      vec[i] = std::thread(write1);
    } else if (i % 5 == 3) {
      vec[i] = std::thread(write2);
    } else {
      vec[i] = std::thread(read);
    }
  }
  for (size_t i = 0; i < 100; i++) {
    vec[i].join();
  }
  return 0;
}