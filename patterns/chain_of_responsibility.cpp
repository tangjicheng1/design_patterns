#include <iostream>
#include <memory>
#include <string>

class Req {
 public:
  Req(const std::string& name, int money) {
    name_ = name;
    money_ = money;
  }

  ~Req() { name_.clear(); }

  std::string GetName() const { return name_; }

  int GetMoney() const { return money_; }

 private:
  std::string name_;
  int money_;
};

class Manager {
 public:
  virtual void Do(const Req& req) = 0;
  void SetNextManager(Manager* manager) { manager_ = manager; }

 protected:
  Manager* manager_;
};

class CommonManager : public Manager {
  void Do(const Req& req) override {
    if (req.GetMoney() < 100) {
      std::cout << req.GetName() << " done by Common Manager " << req.GetMoney() << std::endl;
      return;
    }
    if (manager_ != nullptr) {
      manager_->Do(req);
    } else {
      std::cout << "Error: No next manager can do it. \nNow is in CommonManager.\n";
    }
    return;
  }
};

class MajorManager : public Manager {
  void Do(const Req& req) override {
    if (req.GetMoney() < 1000) {
      std::cout << req.GetName() << " done by Major Manager " << req.GetMoney() << std::endl;
      return;
    }
    if (manager_ != nullptr) {
      manager_->Do(req);
    } else {
      std::cout << "Error.\n";
    }
    return;
  }
};

int main() {
  CommonManager* common_manager = new CommonManager();
  MajorManager* major_manager = new MajorManager();
  common_manager->SetNextManager(major_manager);
  major_manager->SetNextManager(nullptr);

  Manager* m = common_manager;

  Req q1("Alex", 99);
  Req q2("Bob", 786);

  m->Do(q1);
  m->Do(q2);

  delete major_manager;
  delete common_manager;
  return 0;
}