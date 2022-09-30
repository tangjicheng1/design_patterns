#include <iostream>
#include <string>
#include <vector>

std::string operator"" _str(const char* str, std::size_t len) {
  std::string ret(str);
  ret += "_str";
  std::cout << "input:" << str << " // " << len << std::endl;
  return ret;
}

// std::string operator"" _KK(unsigned long long int x) {
//   std::cout << "input: " << x << std::endl;
//   return std::to_string(x) + "K";
// }

std::string operator"" _KK(const char* x) {
  std::cout << "input2: " << x << std::endl;
  return std::string(x) + "K";
}

std::string operator"" _PI(long double x) {
  std::cout << "input: " << x << std::endl;
  return std::to_string(x) + "PI";
}

std::string operator"" _char(char ch) {
  std::cout << "input: " << ch << std::endl;
  std::string ret(1, ch);
  return ret;
}

std::string operator"" _LITERAL(const char* str) {
  std::cout << "input: " << str << std::endl;
  return std::string(str);
}

std::string operator"" _LITERAL(unsigned long long int i) {
  std::cout << "input2: " << i << std::endl;
  return std::string("int_i");
}

void test1() {
  std::cout << "Hello"_str << std::endl;
  std::cout << 12_KK << std::endl;
  std::cout << 12.21_PI << std::endl;
  std::cout << .123_PI << std::endl;
  std::cout << 0e-1 << std::endl;
  std::cout << 0e-1_PI << std::endl;
  std::cout << 'i'_char << std::endl;
  std::cout << 0x123AB_LITERAL << std::endl;
}

int main() {
  test1();
  return 0;
}