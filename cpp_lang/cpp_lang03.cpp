// 静态存储区的大小
#include <iostream>

// 和操作系统相关，BSS字段或DATA字段的大小
// 未初始化对应BSS字段，已初始化对应DATA字段
// https://stackoverflow.com/questions/18371584/what-is-the-maximum-size-of-static-array-that-can-be-declared-in-gcc

constexpr int N = 1024 * 1024 * 1024;  // GB
constexpr int M = 1;

char arr[M][N];

int main() {
  // 局部变量，对应stack大小
  for (int j = 0; j < M; j++) {
    for (int i = 0; i < N; i++) {
      if (arr[j][i] != 0) {
        std::cout << "False" << std::endl;
      }
    }
  }

  std::cout << "Hello" << std::endl;
  return 0;
}