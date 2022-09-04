#include <iostream>
#include <tuple>

struct A {
    private:
    void print(int x) {std::cout << x << "\n";}
};

int main() {
    // Write C++ code here
    std::cout << "Hello world!";
    
    A a;
    std::apply(&A::print, std::tuple<A, int>{a, 10});

    return 0;
}