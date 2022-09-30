#include <iostream>
#include <string>

void test1() {
  std::string s = "Hello World";
  std::cout << s.length() << " " << s.capacity() << std::endl;
  s += "!";
  std::cout << s.length() << " " << s.capacity() << std::endl;

  std::string s1("你好");
  std::string s2("世界");
  std::cout << s1.length() << " " << s1.capacity() << std::endl;
  std::cout << s2.length() << " " << s2.capacity() << std::endl;
  std::cout << s1.compare(s2) << std::endl;
}

int main() {
  test1();
  return 0;
}