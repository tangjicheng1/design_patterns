#include <iostream>
#include <type_traits>

void foo(int i) {
  std::cout << "int:" << i << std::endl;
  return;
}

void foo(char* str) {
  if (str == nullptr) {
    std::cout << "str: nullptr\n";
    return;
  }
  std::cout << "str:" << str << std::endl;
}

void test() {
  // foo(NULL);
  foo(0);
  foo(nullptr);

  std::cout << std::boolalpha << std::is_same<decltype(nullptr), decltype(NULL)>::value << std::endl;
  std::cout << std::boolalpha << std::is_same<long, decltype(NULL)>::value << std::endl;
  std::cout << std::boolalpha << std::is_same<void*, decltype(NULL)>::value << std::endl;
}

int main() {
  test();
  return 0;
}