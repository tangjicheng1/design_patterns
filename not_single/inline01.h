// 测试cpp 17 特性 ：inline变量
#pragma once

// 不加inline，被两个cpp文件include后，链接。会出错，重复定义了变量double a
// double a = 1.0;

// cpp17 新特性，相当于
// static double a = 1.0;
inline double a = 1.0;

// 但是cpp17中，inline，还可以用来修饰类内的静态成员变量
struct A {
  // 如果不用inline修饰，则会出错，因为类内的静态成员变量，不允许在类内进行初始化。
  inline static double b = 2.0;
  static double c;
};

// 以下语句会有重复定义的错误
// double A::c = 3.0;

// 以下语句同样会出错，因为不允许在类外，用static修饰类内静态成员变量
// static double A::c = 3.0;

// 以下语句可以通过编译，这是cpp17的新特性
inline double A::c = 3.0;