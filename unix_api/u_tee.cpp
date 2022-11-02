#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv) {
  if (argc > 2) {
    const char* usage_str = "usage: tee output.txt\n";
    write(STDOUT_FILENO, usage_str, strlen(usage_str));
    exit(1);
    return 0;
  }
  mode_t open_mode = S_IWGRP | S_IWOTH | S_IWUSR | S_IRGRP | S_IROTH | S_IRUSR;
  int fd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC, open_mode);
  if (fd == -1) {
    const char* err_str = "err: cannot open file\n";
    write(STDOUT_FILENO, err_str, strlen(err_str));
    exit(1);
  }

  const int buff_size = 100;
  char buff[buff_size];
  int read_num = 0;
  while ((read_num = read(STDIN_FILENO, buff, buff_size)) >= 0) {
    write(STDOUT_FILENO, buff, read_num);
    write(fd, buff, read_num);
  }

  close(fd);
  return 0;
}