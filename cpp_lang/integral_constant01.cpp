#include <iostream>
#include <type_traits>

template <int n>
struct f : std::integral_constant<int, f<n - 1>::value + 1> {};

template <>
struct f<1> : std::integral_constant<int, 1> {};

int main() {
  constexpr int N = 10;
  std::cout << f<N>::value << std::endl;
  return 0;
}