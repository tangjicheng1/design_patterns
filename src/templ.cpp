// 模板模式
// 在基类定义并且实现接口函数，但是接口函数中的具体步骤延迟到子类中实现
#include <iostream>

class Computer {
 public:
  void product() {
    std::cout << "Computer product is start." << std::endl;
    InstallCpu();
    InstallMem();
    std::cout << "Computer product is end.\n" << std::endl;
  }

 protected:
  virtual void InstallCpu() = 0;
  virtual void InstallMem() = 0;
};

class Hp : public Computer {
 protected:
  void InstallCpu() override { std::cout << "Hp: Install i5 CPU." << std::endl; }
  void InstallMem() override { std::cout << "Hp: Install 16G memory." << std::endl; }
};

class Dell : public Computer {
 protected:
  void InstallCpu() override { std::cout << "Dell: Install i7 CPU." << std::endl; }
  void InstallMem() override { std::cout << "Dell: Install 8G memory." << std::endl; }
};

int main() {
  Computer* c1 = new Hp();
  c1->product();
  Computer* c2 = new Dell();
  c2->product();
  return 0;
}
