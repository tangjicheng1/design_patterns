#include "json.hpp"
#include <iostream>

using json = nlohmann::json;

void test1() {
  json j = {{"one", 1}, {"two",2}};
  std::cout << j.size() << std::endl;
}

int main() {
  test1();
  return 0;
}