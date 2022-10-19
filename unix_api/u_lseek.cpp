#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>

inline void err_exit(const char* str) {
  printf("Error: %s\n", str);
  exit(1);
}

#define CHECK(x)                            \
  do {                                      \
    if ((x) != true) {                      \
      printf("%s:%d ", __FILE__, __LINE__); \
      err_exit(#x);                         \
    }                                       \
  } while (0)

int main(int argc, char** argv) {
  if (argc <= 2) {
    printf("usage: %s commands\n", argv[0]);
    return 0;
  }

  int fd = open(argv[1], O_RDWR | O_CREAT);
  CHECK(fd != -1);
  char command_type = 0;
  for (int i = 2; i < argc; i++) {
    if (strlen(argv[i]) < 2) {
      printf("Error: illegal command\n");
      exit(1);
    }
    command_type = argv[i][0];
    switch (command_type) {
      case 'r': {
        ssize_t read_size = (ssize_t)strtol(&argv[i][1], NULL, 10);
        read_size = std::min(read_size, (long)100);
        char* buff = (char*)malloc((read_size + 1) * sizeof(char));
        ssize_t read_num = read(fd, buff, read_size);
        CHECK(read_num >= 0);
        buff[read_num] = '\0';
        printf("read: %s\n", buff);
        free(buff);
        break;
      }

      case 'w': {
        ssize_t write_num = write(fd, &argv[i][1], strlen(&argv[i][1]));
        CHECK(write_num >= 0);
        break;
      }

      case 's': {
        int offset = (int)strtol(&argv[i][1], NULL, 10);
        CHECK(lseek(fd, offset, SEEK_SET) != -1);
        break;
      }

      default: {
        printf("Error: illegal command\n");
        CHECK(close(fd) == 0);
        exit(1);
        break;
      }
    }
  }
  CHECK(close(fd) == 0);
  return 0;
}