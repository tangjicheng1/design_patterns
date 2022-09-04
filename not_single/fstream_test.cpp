#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void test1() {
  std::ofstream ofs("input.txt", std::ios::app);
  ofs << "Hello\n";
}

void test2() {
  std::ofstream ofs("input.txt");
  ofs << "Hello without app\n";
}

void test3() {
  std::ofstream ofs("output.bin", std::ios::binary);
  float x = 1.234;
  ofs << "Hello\n" << 1 << x << 1.0 << "end\n";
}

int main() {
  // test1();
  // test2();
  test3();
  return 0;
}