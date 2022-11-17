#include <iostream>
#include <math.h>

int computeArea(int ax1, int ay1, int ax2, int ay2, int bx1, int by1, int bx2, int by2) {
  if (ay2 <= by1 || by2 <= ay1) {
    return 0;
  }
  if (ax2 <= bx1 || bx2 <= ax1) {
    return 0;
  }
  return std::min(abs(ay1 - by2), abs(ay2 - by1)) * std::min(abs(ax2 - bx1), abs(ax1 - bx2));
}