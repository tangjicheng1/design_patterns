#pragma once

// 链接错误，重复定义 var_without_extern
// int var_without_extern;

// 内部链接，在两份cpp文件中，分别定义了两个var_with_static，两者互不干扰
static int var_with_static = 0;

// 声明
extern int var_with_extern;

