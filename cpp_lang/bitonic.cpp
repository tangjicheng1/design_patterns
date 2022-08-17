#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

struct KV {
  float key;
  int index;
};

// 2^k
// data[begin, end] 已经是双调序列，排序成为有序的序列
void bitonic_sort(std::vector<KV>& data, int begin, int end, bool larger = true) {
  if (end - begin < 1) {
    return;
  }
  int length = (end - begin + 1) / 2;
  int mid = begin + length;

  // 分为两个序列：[begin, mid - 1], [mid, end]
  // 这个for循环可以完全并行执行
  if (larger) {
    // 从小到大排序
    for (int i = begin; i < mid; i++) {
      if (data[i].key > data[i + length].key) {
        KV temp;
        temp.key = data[i].key;
        temp.index = data[i].index;
        data[i] = data[i + length];
        data[i + length] = temp;
      }
    }
  } else {
    // 从大到小排序
    for (int i = begin; i < mid; i++) {
      if (data[i].key < data[i + length].key) {
        KV temp;
        temp.key = data[i].key;
        temp.index = data[i].index;
        data[i] = data[i + length];
        data[i + length] = temp;
      }
    }
  }

  bitonic_sort(data, begin, mid - 1, larger);
  bitonic_sort(data, mid, end, larger);
}

// 构造双调序列
void build_bitonic(std::vector<KV>& data, int begin, int end) {
  int step = 1;
  int length = end - begin + 1;
  while (step <= length / 2) {
    bool larger = true;
    // 这里的for循环可以完全并行
    for (int i = 0; i < length; i += step) {
      bitonic_sort(data, i, i + step - 1, larger);
      larger = !larger;
    }
    step *= 2;
  }
}

bool is_sorted(const std::vector<KV>& data, bool larger = true) {
  if (larger) {
    for (int i = 0; i < data.size() - 1; i++) {
      if (data[i].key > data[i + 1].key) {
        return false;
      }
    }
    return true;
  }
  for (int i = 0; i < data.size() - 1; i++) {
    if (data[i].key < data[i + 1].key) {
      return false;
    }
  }
  return true;
}

void test2(int argc, char** argv) {
  int N = 16;
  if (argc == 2) {
    int input_n = atoi(argv[1]);
    N = input_n;
  }
  srand((unsigned int)time(NULL));
  std::vector<KV> data;
  data.reserve(N);
  for (int i = 0; i < N; i++) {
    KV temp;
    temp.key = (float)(rand() % 100);
    temp.index = i;
    data.push_back(temp);
  }

  for (auto iter : data) {
    std::cout << iter.key << " " << iter.index << std::endl;
  }

  build_bitonic(data, 0, N - 1);
  bitonic_sort(data, 0, N - 1, true);

  std::cout << "After sort:\n";
  for (auto iter : data) {
    std::cout << iter.key << " " << iter.index << std::endl;
  }
  bool is_ok = is_sorted(data);
  std::cout << std::boolalpha << is_ok << std::endl;
}

void test1() {
  std::vector<KV> data;
  for (int i = 0; i < 4; i++) {
    KV temp;
    temp.key = i + 1;
    temp.index = i;
    data.push_back(temp);
  }
  for (int i = 0; i < 4; i++) {
    KV temp;
    temp.key = 10 - i;
    temp.index = i + 4;
    data.push_back(temp);
  }
  for (auto iter : data) {
    std::cout << iter.key << "," << iter.index << std::endl;
  }
  std::cout << "After sort:\n";

  std::vector<KV> data2(data);

  bitonic_sort(data, 0, 7);
  for (auto iter : data) {
    std::cout << iter.key << "," << iter.index << std::endl;
  }
  bitonic_sort(data2, 0, 7, false);
  std::cout << "big --> small\n";
  for (auto iter : data2) {
    std::cout << iter.key << "," << iter.index << std::endl;
  }
}

int main(int argc, char** argv) {
  // test1();
  test2(argc, argv);
  return 0;
}