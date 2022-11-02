#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("usage: %s filename\n", argv[0]);
    return 0;
  }
  mode_t file_mode = S_IWGRP | S_IWUSR | S_IRGRP | S_IROTH | S_IRUSR;
  int fd = open(argv[1], O_WRONLY | O_TRUNC | O_CREAT, file_mode);
  if (fd == -1) {
    printf("Error: cannot touch file %s\n", argv[1]);
    return 1;
  }
  if (close(fd) == -1) {
    printf("Error: close file %s\n", argv[1]);
    return 1;
  }

  return 0;
}