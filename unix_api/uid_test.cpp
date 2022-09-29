#include <unistd.h>
#include <stdio.h>
#include <iostream>

int main() {
  std::cout << "uid: " << getuid() << " gid: " << getgid() << std::endl;
  return 0;
}