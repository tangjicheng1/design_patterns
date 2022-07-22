#include <array>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <tuple>
#include <vector>

struct A {
  int a;
  double b;
  std::string s;
  void print() { std::cout << a << " , " << b << " , " << s << std::endl; }
};

void test01() {
  A a_struct{1, 2.0, "hello"};
  // []内必须是全部的struct的成员变量
  auto& [x, y, str] = a_struct;
  x = 4;
  a_struct.print();
}

void test02() {
  // 使用 结构化绑定 遍历map
  // 因为map的元素，是std::pair
  std::map<std::string, int> m;
  for (int i = 0; i < 10; i++) {
    m[std::string("key") + std::to_string(i)] = i;
  }
  for (const auto& [key, value] : m) {
    std::cout << key << " , " << value << std::endl;
    // key是std::string类型，key.size()查看字符串长度
    // std::cout << "key_len: " << key.size() << std::endl;
  }
  std::cout << "origin use pair\n";
  for (std::pair<std::string, int> iter : m) {
    std::cout << "std::pair : " << iter.first << " , " << iter.second << std::endl;
  }
}

struct B {
  double x;
  double y;
};

B getB(int val) {
  B b;
  b.x = double(val / 10);
  b.y = double(val % 10);
  return b;
}

void test03() {
  auto [x, y] = getB(54);
  std::cout << x << " , " << y << std::endl;
}

struct C {
  C() { std::cout << "C()\n"; }
  C(const C& c) {
    x = c.x;
    y = c.y;
    std::cout << "C(const C&)\n";
  }
  ~C() { std::cout << "~C()\n"; }

  int x;
  int y;

  static int s_x;
  const static int cs_x = 1;
};

int C::s_x = 10;

void test04() {
  C c;
  c.x = 10;
  c.y = 3;

  auto [x1, y1] = c;
  { auto [x2, y2] = c; }

  std::cout << c.x << " , " << c.y << std::endl;
  std::cout << x1 << " , " << y1 << std::endl;
  std::cout << __cplusplus << "  end test04\n";
}

void test05() {
  std::array<int, 2> std_arr{3, 5};
  std::vector<std::array<int, 2>> vec;
  for (int i = 0; i < 4; i++) {
    vec.push_back(std::array<int, 2>{i, 10 * i});
  }
  std::cout << "origin:\n";
  for (auto iter : vec) {
    std::cout << iter[0] << " , " << iter[1] << std::endl;
  }
  for (auto& iter : vec) {
    auto& [x, y] = iter;
    x += 1;
    y += 2;
  }
  std::cout << "now:\n";
  for (auto iter : vec) {
    std::cout << iter[0] << " , " << iter[1] << std::endl;
  }
}

void test06() {
  std::array<int, 2> std_arr{3, 5};
  auto [x, y] = std_arr;
  std::cout << x << " , " << y << std::endl;
  std::tie(y, x) = std_arr;
  std::cout << x << " , " << y << std::endl;
  std::tuple<int, int> tup{1, 3};
  tup = std_arr;
  std::cout << "tuple" << std::endl;
  std::cout << std::get<0>(tup) << " , " << std::get<1>(tup) << std::endl;
}

int main() {
  // test01();
  // test02();
  // test03();
  // test04();
  // test05();
  test06();
  return 0;
}