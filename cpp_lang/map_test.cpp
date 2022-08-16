#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <utility>

void test1() {
  std::map<std::string, int> map_1;
  std::map<std::string, int> map_2{{"123", 1}, {"abc", 3}};
  map_1["123"] = 4;
  map_2["def"] = 2;
  std::cout << std::boolalpha << map_1.empty() << " " << map_1.size() << " " << map_1.max_size() << std::endl;
  map_1.clear();
  map_1.insert({"xx", 1});
  map_1.insert(map_1.begin(), {"yy", 2});
  map_1.insert(map_2.begin(), map_2.end());
}

void test2() {
  std::unordered_map<std::string, std::string> map_1;
  std::unordered_map<std::string, std::string> map_2{{"123", "1"}, {"abc", "3"}};
  map_1["123"] = "4";
  map_2["def"] = "2";
  std::cout << std::boolalpha << map_1.empty() << " " << map_1.size() << " " << map_1.max_size() << std::endl;
  auto iter1 = map_1.find("123");
  map_1.insert(iter1, {"xx", "1"});
  for (auto pair : map_1) {
    std::cout << pair.first << " " << pair.second << std::endl;
  }
}

int main() {
  // test1();
  test2();
  return 0;
}