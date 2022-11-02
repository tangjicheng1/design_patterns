#if defined(__GNUC__) && defined(__linux)
#  include <gnu/libc-version.h>

#  include <iostream>
int test(int argc, char** argv) {
  std::cout << "GNU GLIBC version: " << gnu_get_libc_version() << std::endl;
  std::cout << "GNU GLIBC release: " << gnu_get_libc_release() << std::endl;

#  if defined(__linux)
  std::cout << "Linux: " << __linux << std::endl;
#  endif

  return 0;
}

#endif

int main(int argc, char** argv) {
#if defined(__GNUC__) && defined(__linux)
  return test(argc, argv);
#else
  return 0;
#endif
}
