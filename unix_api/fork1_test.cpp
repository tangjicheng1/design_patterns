#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
  exit(0); // for ci
  pid_t pid;
  char buff[1024];
  int status;
  printf("MYSHELL: ");
  while (fgets(buff, 1024, stdin) != NULL) {
    if (buff[strlen(buff) - 1] == '\n') {
      buff[strlen(buff) - 1] = '\0';
    }
    if ((pid = fork()) < 0) {
      printf("fork error\n");
      exit(0);
    }
    // in child process
    if (pid == 0) {
      execlp(buff, buff, (char*)0);
      printf("can not run %s\n", buff);
      exit(127);
    }

    if ((pid = waitpid(pid, &status, 0)) < 0) {
      printf("wait pid error\n");
    }
    printf("MYSHELL: ");
  }
  return 0;
}