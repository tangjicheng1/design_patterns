// 静态存储区的大小
#include <iostream>

constexpr int N = 1024 * 1024 * 1024;  // GB
constexpr int M = 1024;

char arr[M][N];

int main() {
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