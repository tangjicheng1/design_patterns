#include <iostream>
#include <string>
#include <regex>

void test1() {
  std::regex html("<.*>.*</.*>");
  bool ret = std::regex_match("<html>val</html>", html);
  std::cout << std::boolalpha << ret << std::endl;
  std::cout << std::boolalpha << std::regex_match("<xml>val<xml>", html) << std::endl;
  std::cmatch m;
  ret = std::regex_search("aaa<html>val</html>", m, html);
  std::cout << m.str() << std::endl << m.position() << std::endl;
}

int main() {
  test1();
  return 0;
}