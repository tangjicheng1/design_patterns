// 策略模式
#include <iostream>

class StrategyInterface {
 public:
  StrategyInterface() = default;
  virtual ~StrategyInterface() = default;
  virtual void DoStrategy() = 0;
};

class Strategy1 : public StrategyInterface {
 public:
  void DoStrategy() override { std::cout << "S1 strategy is doing...\n"; }
};

class Strategy2 : public StrategyInterface {
 public:
  void DoStrategy() override { std::cout << "S2 strategy is doing...\n"; }
};

class Context {
 public:
  Context(StrategyInterface* strategy) : strategy_(strategy) {}
  virtual ~Context() { delete strategy_; }
  void ExecuteStrategy() { strategy_->DoStrategy(); }

 private:
  StrategyInterface* strategy_;
};

int main() {
  Context c1(new Strategy1());
  c1.ExecuteStrategy();

  Context c2(new Strategy2());
  c2.ExecuteStrategy();

  return 0;
}