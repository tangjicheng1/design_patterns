#include <errno.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

int main(int argc, char** argv) {
  printf("EACCES: %s\n", strerror(EACCES));
  errno = EACCES;
  perror(argv[0]);
  return 0;
}