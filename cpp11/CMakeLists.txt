cmake_minimum_required(VERSION 3.0)
project(cpp17 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 11)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/cpp11)

function(add_cpp11 filename)
add_executable(${filename} ${SRC_DIR}/${filename}.cpp)
endfunction()

add_cpp11(tuple01)
add_cpp11(integral_constant01)