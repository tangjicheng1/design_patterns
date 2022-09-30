// 表驱动模式
#include <iostream>
#include <map>
#include <string>

// switch-case or if-else
int main_1() {
  std::string str;
  int x = 0;
  if (str == "one") {
    x += 1;
  } else if (str == "two") {
    x += 2;
  } else if (str == "three") {
    x += 3;
  } else {
    x += 0;
  }
  return 0;
}

int main_2() {
  std::map<std::string, int> table_driven({{"one", 1}, {"two", 2}, {"three", 3}});
  std::string str;
  int x = 0;
  if (table_driven.find(str) != table_driven.end()) {
    x += table_driven[str];
  }

  return 0;
}

int main() {
  main_1();
  main_2();
  return 0;
}