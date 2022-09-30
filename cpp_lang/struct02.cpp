// 结构化绑定，对一般的类进行设置绑定。
// 包含私有成员变量，需要绑定成为tuple like
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

class Consumer {
 public:
  Consumer(int id, double money, const std::string& name) : id_(id), money_(money), name_(name) {}
  Consumer() = default;
  virtual ~Consumer() = default;

  int GetId() { return id_; }

  double GetMoney() { return money_; }

  std::string GetName() { return name_; }

 private:
  int id_ = 0;
  double money_ = 0.0;
  std::string name_;
};

template <>
struct std::tuple_size<Consumer> {
  static constexpr size_t value = 3;
};



void test01() {}

int main() {
  test01();
  return 0;
}