#include <deque>
#include <stack>
#include <queue>
#include <vector>
#include <iostream>

void test1() { 
  std::deque<double> a{1, 2, 3, 4, 5, 6};
  std::stack<double> stack_1(a); 
  std::cout << a.size() << std::endl;
  std::queue<double> queue_1(std::move(a));
  std::cout << a.size() << std::endl;

  int stack_size = stack_1.size();
  for (int i = 0; i < stack_size; i++) {
    std::cout << stack_1.top() << " ";
    stack_1.pop();
  }
  std::cout << std::endl;

  int queue_size = queue_1.size();
  for (int i = 0; i < queue_size; i++) {
    std::cout << queue_1.front() << " ";
    queue_1.pop();
  }
  std::cout << std::endl;
}

int main() {
  test1();
  return 0;
}