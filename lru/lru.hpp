#pragma once

#include <stdio.h>
#include <functional>  // for std::hash
#include <list>
#include <unordered_map>
#include <vector>

// hash fucntion for std::vector<cpp_basic_type>, just like std::vector<float>/std::vector<int>/...
// a magic number for hash, 0x9e3779b9, reference:
// https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
template <typename T>
struct VectorHash {
  std::size_t operator()(const std::vector<T>& vec) const {
    std::size_t seed = vec.size();
    for (const auto& iter : vec) {
      seed ^= std::hash<T>()(iter) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

template <typename Key, typename T, typename Hash = std::hash<Key>>
class LruCache {
 public:
  explicit LruCache(std::size_t max_size) : max_size_(max_size) {}
  void Insert(const Key& key, const T& value) {
    auto iter = map_.find(key);
    if (iter != map_.end()) {
      iter->second.value = value;
      MoveToFront(iter->second.lru_iterator);
      return;
    }

    while (Size() + 1 > max_size_) {
      map_.erase(lru_list_.back());
      lru_list_.pop_back();
    }

    lru_list_.emplace_front(key);
    map_.emplace(key, ValueType{value, lru_list_.begin()});
    return;
  }

  T& At(const Key& key) {
    auto iter = map_.find(key);
    if (iter == map_.end()) {
      printf("Error: LruCache cannot find key\n");
      abort();
    }
    MoveToFront(iter->second.lru_iterator);
    return iter->second.value;
  }

  bool Contains(const Key& key) const { return map_.find(key) != map_.end(); }

  std::size_t Size() const { return map_.size(); }

  void Clear() {
    map_.clear();
    lru_list_.clear();
    return;
  }

 private:
  using IteratorType = typename std::list<Key>::iterator;
  struct ValueType {
    T value;
    IteratorType lru_iterator;
  };

  void MoveToFront(IteratorType iter) {
    lru_list_.splice(lru_list_.begin(), lru_list_, iter);
    return;
  }

  std::size_t max_size_;
  std::unordered_map<Key, ValueType, Hash> map_;
  std::list<Key> lru_list_;
};