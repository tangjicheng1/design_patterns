#include <iostream>
#include <cstdio>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>
#include <functional>

std::queue<int> fifo_queue;
std::mutex mutex_for_queue;
std::condition_variable cv;

bool stop = false;

int produce_func() {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return rand() % 100;
}

int consume_func(int x) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  return 0;
}

void producer() {
  while (true) {
    std::unique_lock<std::mutex> lock_for_queue(mutex_for_queue);
    cv.wait(lock_for_queue, []() { return (fifo_queue.size() < 10); });
    int produce = produce_func();
    printf("[Producer] %d\n", produce);
    fifo_queue.push(produce);
    lock_for_queue.unlock();
    cv.notify_one();
    if(stop) {
      break;
    }
  }
}

void consumer() {
  while (true) {
    std::unique_lock<std::mutex> lock_for_queue(mutex_for_queue);
    cv.wait(lock_for_queue, []() { return (fifo_queue.size() > 0); });
    int consume = fifo_queue.front();
    consume_func(consume);
    printf("[Consumer] %d\n", consume);
    fifo_queue.pop();
    lock_for_queue.unlock();
    cv.notify_one();
    if(stop) {
      break;
    }
  }
}

void test1() {
  int N = 32;
  std::vector<std::thread> p;
  std::vector<std::thread> c;

  for (int i = 0; i < N; i++) {
    p.push_back(std::thread(producer));
    c.push_back(std::thread(consumer));
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));
  stop = true;

  std::for_each(p.begin(), p.end(), std::mem_fn(&std::thread::join));
  std::for_each(c.begin(), c.end(), std::mem_fn(&std::thread::join));
}

int main() {
  srand((unsigned)time(NULL));
  test1();
  return 0;
}