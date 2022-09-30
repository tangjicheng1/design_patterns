#include <iostream>
#include <queue>

void test1() {
  auto comp = [](double a, double b) { return a > b; };
  std::priority_queue<double, std::vector<double>, decltype(comp)> pq(comp);
}

int main() {
  test1();
  return 0;
}