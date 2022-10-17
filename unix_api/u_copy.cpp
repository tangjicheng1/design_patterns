#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#define BUFFER_SIZE 1024

void my_copy(const char* input, const char* output) {
  int input_fd = open(input, O_RDONLY);
  if (input_fd == -1) {
    printf("[Error] cannot open file %s\n", input);
    perror("Error");
    exit(1);
  }
  mode_t output_mode = S_IRUSR | S_IRGRP | S_IROTH | S_IWGRP | S_IWOTH | S_IWUSR;
  int output_fd = open(output, O_CREAT | O_WRONLY | O_TRUNC, output_mode);
  if (output_fd == -1) {
    printf("[Error] cannot open file %s\n", output);
  }

  char buffer[BUFFER_SIZE];
  ssize_t read_count = 0;
  while ((read_count = read(input_fd, buffer, BUFFER_SIZE)) > 0) {
    if (write(output_fd, buffer, read_count) == -1) {
      printf("write error\n");
      close(input_fd);
      close(output_fd);
      exit(1);
    }
  }

  if (close(input_fd) == -1) {
    printf("close input error\n");
  }
  if (close(output_fd) == -1) {
    printf("close output error\n");
  }

  if (read_count == -1) {
    printf("read error\n");
    exit(1);
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("usages: %s input_file output_file\n", argv[0]);
    return 1;
  }

  my_copy(argv[1], argv[2]);

  return 0;
}