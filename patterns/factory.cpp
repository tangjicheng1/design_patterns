// 工厂模式，主要的思想是隐藏对象创建的过程
// 通过工厂类来创建具体的产品对象

#include <iostream>

enum Animal { kCat = 0, kDog };

class AnimalBase {
 public:
  virtual void Print() = 0;
  virtual ~AnimalBase() = default;
};

class Cat : public AnimalBase {
 public:
  void Print() override { std::cout << "This is cat." << std::endl; }
};

class Dog : public AnimalBase {
 public:
  void Print() override { std::cout << "This is dog." << std::endl; }
};

class AnimalFactory {
 public:
  AnimalBase* Create(Animal animal) {
    AnimalBase* result = nullptr;
    switch (animal) {
      case kCat: {
        result = new Cat();
        break;
      }
      case kDog: {
        result = new Dog();
        break;
      }
      default: {
        result = nullptr;
        break;
      }
    }
    return result;
  }
};

int main() {
  AnimalFactory* factory = new AnimalFactory();
  AnimalBase* my_cat = factory->Create(Animal::kCat);
  AnimalBase* my_dog = factory->Create(Animal::kDog);
  my_cat->Print();
  my_dog->Print();
  delete my_cat;
  delete my_dog;
  delete factory;
  return 0;
}