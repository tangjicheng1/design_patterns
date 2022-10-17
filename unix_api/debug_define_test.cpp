#include <unistd.h>
#include <iostream>

int main() {
#if defined(DEBUG)
  std::cout << "defined DEBUG" << std::endl;
#else
  std::cout << "NO" << std::endl;
#endif
  return 0;
}
