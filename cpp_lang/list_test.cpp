#include <iostream>
#include <list>
#include <vector>

void test1() {
  std::list<double> a{1, 2, 3, 4, 5, 6};
  std::list<double> b(a.begin(), a.end());
  std::list<double> c(b);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  c.push_back(7.0);
  c.push_front(0.0);
  std::cout << c.front() << " " << c.back() << std::endl;
  c.assign(a.begin(), a.end());
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  c.assign(7, 1.0);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  c.insert(++c.begin(), -1);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  std::cout << c.size() << std::endl;
  auto iter = c.erase(c.begin());
  c.erase(iter);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  c.unique();
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  c.assign(a.begin(), a.end());
  auto comp = [](int a, int b) { return a > b; };
  c.sort(comp);
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  c.reverse();
  for (auto i : c) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  std::vector<int> v(c.begin(), c.end());
  for (auto i : v) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  // v.reverse(); // compile error, vector do not have reverse method

  std::list<double> d(v.begin(), v.end());
  for (auto i : d) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  d.remove(3);
  for (auto i : d) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  d.remove(100);
  for (auto i : d) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  auto filter = [](int a) -> bool { return a > 3; };
  d.remove_if(filter);
  for (auto i : d) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  auto iter2 = ++d.begin();
  d.splice(iter2, c, c.cbegin(), c.cend());
  for (auto i : d) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  d.sort(comp);
  a.sort(comp);
  for (auto i : d) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  for (auto i : a) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  d.merge(a, comp);
  for (auto i : d) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  std::cout << a.size() << " " << c.size() << std::endl;
}

int main() {
  test1();
  return 0;
}