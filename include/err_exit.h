#pragma once

#include <stdio.h>
#include <stdlib.h>

inline void err_exit(const char* str) {
  printf("Error: %s\n", str);
  exit(1);
}