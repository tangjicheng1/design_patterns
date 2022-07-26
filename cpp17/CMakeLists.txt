cmake_minimum_required(VERSION 3.0)
project(cpp17 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/cpp17)

function(add_cpp17 filename)
add_executable(${filename} ${SRC_DIR}/${filename}.cpp)
endfunction()

add_executable(inline_test ${SRC_DIR}/inline01.cpp ${SRC_DIR}/inline02.cpp)

add_cpp17(struct01)
