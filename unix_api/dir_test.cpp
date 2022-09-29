#include "apue.h"

#include <dirent.h>
#include <stdio.h>

int main(int argc, char** argv) {
  DIR* d;
  struct dirent* dirp;
  if (argc != 2) {
    printf("usage: %s dir\n", argv[0]);
    return 0;
  }
  if ((d = opendir(argv[1])) == NULL) {
    printf("Error: cannot open %s\n", argv[1]);
    return 0;
  }
  while((dirp = readdir(d)) != NULL) {
    printf("DIR: %s\n", dirp->d_name);
  }
  return 0;
}