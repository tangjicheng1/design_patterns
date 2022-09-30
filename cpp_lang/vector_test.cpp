#include <iostream>
#include <vector>

struct A {
  A(double val_, int num_) : val(val_), num(num_) {}
  double val;
  int num;
};

void test1() {
  std::vector<double> v1(10, 1.0f);
  std::vector<float> v2({1, 2, 3, 4});
  std::vector<int> v3(v2.begin(), v2.end() - 1);

  std::cout << "v1: ";
  for (auto i : v1) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  std::cout << "v2: ";
  for (std::vector<float>::iterator iter = v2.begin(); iter != v2.end(); ++iter) {
    std::cout << *iter << " ";
  }
  std::cout << std::endl;

  std::cout << "v3: ";
  for (auto i : v3) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  std::cout << "v3.front: " << v3.front() << " v3.back: " << v3.back() << std::endl;
  std::cout << "v3.size: " << v3.size() << " v3.capacity: " << v3.capacity() << std::endl;
  v3.reserve(10);
  std::cout << "v3.size: " << v3.size() << " v3.capacity: " << v3.capacity() << std::endl;
  v3.shrink_to_fit();
  std::cout << "v3.size: " << v3.size() << " v3.capacity: " << v3.capacity() << std::endl;
  v3.reserve(1);
  std::cout << "v3.size: " << v3.size() << " v3.capacity: " << v3.capacity() << std::endl;

  for (int i = 0; i < 10; ++i) {
    v3.push_back(i);
  }

  for (auto i : v3) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  std::vector<int>::iterator iter = v3.begin();
  for (int i = 0; i < v3.size(); ++i) {
    if (i > 4) {
      iter = v3.erase(iter);
    }
    std::cout << i << " v3.size: " << v3.size() << std::endl;
  }
  for (auto i : v3) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  auto iter2 = v3.begin();
  auto iter3 = v3.begin() + 4;
  v3.erase(iter2, iter3);
  for (auto i : v3) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  v3.resize(3);
  for (auto i : v3) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  v3.resize(10, -1);
  for (auto i : v3) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  v3.emplace_back(1);
  for (auto i : v3) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  v3.emplace(v3.begin(), 100);
  for (auto i : v3) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  std::vector<A> v4;
  v4.emplace_back(1.0, 1);
  v4.emplace(v4.begin(), 2.0, 2);
  for (auto iter : v4) {
    std::cout << iter.val << " " << iter.num << std::endl;
  }

}

int main() {
  test1();
  return 0;
}