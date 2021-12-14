// 单例模式
// 主要需要保证两点：
// 1. 保证只有一个对象
// 2. 提供全局的接口

#include <iostream>

// lazy版本，在第一次调用的时候生成实例
class SingletonLazy {
 public:
  static SingletonLazy* GetInstance();
  static void DelInstance();

  int func();

  SingletonLazy(const SingletonLazy& obj) = delete;
  SingletonLazy operator=(const SingletonLazy& obj) = delete;
  ~SingletonLazy();

 private:
  SingletonLazy();

 private:
  static SingletonLazy* ptr_lazy_;
  int count;
};

SingletonLazy* SingletonLazy::ptr_lazy_ = nullptr;

SingletonLazy* SingletonLazy::GetInstance() {
  if (ptr_lazy_ == nullptr) {
    ptr_lazy_ = new SingletonLazy();
  }
  // 考虑到线程安全，上述if代码段需要加锁
  return ptr_lazy_;
}

void SingletonLazy::DelInstance() {
  auto del_ptr = ptr_lazy_;
  ptr_lazy_ = nullptr;
  delete del_ptr;
  return;
}

int SingletonLazy::func() {
  std::cout << "Call func() " << std::endl;
  count++;
  std::cout << "Now count is " << count << std::endl;
  return count;
}

SingletonLazy::~SingletonLazy() {}

SingletonLazy::SingletonLazy() { this->count = 0; }

int lazy_main() {
  SingletonLazy::GetInstance()->func();
  SingletonLazy::GetInstance()->func();
  SingletonLazy::DelInstance();
  SingletonLazy::GetInstance()->func();
  return 0;
}

// eager版本，在进入main函数之前，定义
// 只需要将上述版本的ptr_lazy_初始化为实例的指针
// SingletonLazy* SingletonLazy::ptr_lazy_ = nullptr; // Lazy版本
// SingletonLazy* SingletonLazy::ptr_lazy_ = new SingletonLazy(); // Eager版本

// 上述两个版本的对象实例在内存的堆上，另外一种常用的方式是用全局静态变量和模板
template <typename T>
class singleton {
 public:
  static T* Instance() {
    static T obj;
    return &obj;
  }
};

class A : public singleton<A> {
 public:
  void Print() {
    std::cout << "Call Print() " << std::endl;
    std::cout << "count: " << count << std::endl;
    count++;
    return;
  }
  A() { count = 0; }
  ~A() {}

 private:
    int count;
};

int template_singleton() {
  A::Instance()->Print();
  A::Instance()->Print();
  return 0;
}

int main() {
  lazy_main();
  template_singleton();
  return 0;
}