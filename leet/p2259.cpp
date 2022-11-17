#include <iostream>
#include <string>
using namespace std;

string removeDigit(string number, char digit) {
  bool find_in_forward_loop = false;
  size_t remove_pos = 0;
  size_t last_digit_pos = 0;
  for (size_t i = 0; i < number.size() - 1; i++) {
    if (number[i] == digit) {
      last_digit_pos = i;
      if (number[i] < number[i + 1]) {
        find_in_forward_loop = true;
        remove_pos = i;
        break;
      }
    }
  }
  if (find_in_forward_loop) {
    return number.erase(remove_pos, 1);
  }
  if (number[number.size() - 1] == digit) {
    return number.erase(number.size() - 1, 1);
  }
  return number.erase(last_digit_pos, 1);
}

int main() {
  return 0;
}