#include <cstdlib>
#include <ctime>
// #include <format>
#include <iostream>
#include <list>
#include <string>

class NumberGenerator;

class Observer {
 public:
  virtual void update(NumberGenerator* generator) = 0;
};

class NumberGenerator {
 public:
  virtual ~NumberGenerator() = default;
  virtual int Gen() = 0;
  int GetStatus() { return status_; }
  void Notify() {
    for (auto iter : observer_list_) {
      iter->update(this);
    }
  }

  std::list<Observer*> GetObservers() { return observer_list_; }

 protected:
  int status_;
  std::list<Observer*> observer_list_;
};

class PrintObserver : public Observer {
 public:
  void update(NumberGenerator* generator) override { this->status_ = generator->GetStatus(); }
  void Print() { std::cout << "RandObserver status is " << status_ << std::endl; }

 private:
  int status_;
};

class RandGenerator : public NumberGenerator {
 public:
  RandGenerator() {
    observer_list_.push_back(new PrintObserver());
    srand((unsigned)time(NULL));
  }
  ~RandGenerator() {
    for (auto iter : observer_list_) {
      PrintObserver* obs = static_cast<PrintObserver*>(iter);
      delete obs;
    }
    observer_list_.clear();
  }
  int Gen() {
    status_ = rand() % 100;
    this->Notify();
    return status_;
  }
};

void test1() {
  RandGenerator rand_gen;
  rand_gen.Gen();
  auto list_gen = rand_gen.GetObservers();
  PrintObserver* print_obs = static_cast<PrintObserver*>(*(list_gen.begin()));
  print_obs->Print();
}

int main() {
  test1();
  return 0;
}