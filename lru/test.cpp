#include "lru.hpp"

#include <iostream>
#include <string>

void test1() {
  LruCache<std::string, std::string> lru_cache(2);
  lru_cache.Insert(std::string("Hello"), std::string("world"));
  lru_cache.Insert(std::string("1"), std::string("11"));
  lru_cache.Insert(std::string("2"), std::string("22"));

  std::cout << std::boolalpha << "Hello: " << lru_cache.Contains(std::string("Hello")) << std::endl;
  std::cout << std::boolalpha << "1: " << lru_cache.Contains(std::string("1")) << std::endl;
  std::cout << std::boolalpha << "2: " << lru_cache.Contains(std::string("2")) << std::endl;

  return;
}

void test2() {
  LruCache<std::vector<double>, int, VectorHash<double>> lru_cache(2);
  lru_cache.Insert(std::vector<double>{1, 2, 3}, 1);
  lru_cache.Insert(std::vector<double>{1, 2, 4}, 1);
  lru_cache.Insert(std::vector<double>{2, 2, 4}, 1);
  std::cout << std::boolalpha << "0: " << lru_cache.Contains(std::vector<double>{1, 2, 3}) << std::endl;
  std::cout << std::boolalpha << "1: " << lru_cache.Contains(std::vector<double>{1, 2, 4}) << std::endl;
  std::cout << std::boolalpha << "2: " << lru_cache.Contains(std::vector<double>{2, 2, 4}) << std::endl;
  return;
}

int main() {
  // test1();
  test2();
  return 0;
}