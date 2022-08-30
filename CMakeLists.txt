cmake_minimum_required(VERSION 3.0)
project(design_patterns LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

enable_testing()

set(ROOT_DIR ${CMAKE_SOURCE_DIR})
include_directories(${ROOT_DIR}/include)

message(STATUS "C++ STANDARD: ${CMAKE_CXX_STANDARD}")

function(add_each dir_name)
file(GLOB cur_files ${dir_name}/*.cpp)
foreach(cur_file ${cur_files})
  string(REPLACE ".cpp" "" cur_exe_file ${cur_file})
  string(REPLACE "/" ";" cur_exe_file_list ${cur_exe_file})
  list(LENGTH cur_exe_file_list list_len)
  math(EXPR list_last "${list_len} - 1")
  list(GET cur_exe_file_list ${list_last} cur_exe)
  # message(STATUS "Added ${cur_exe}")
  add_executable(${cur_exe} ${cur_file})
  add_test(NAME ${cur_exe} COMMAND ${cur_exe})
endforeach()
endfunction()

add_each(${ROOT_DIR}/patterns)
add_each(${ROOT_DIR}/cpp_lang)
add_each(${ROOT_DIR}/cpp11)
add_each(${ROOT_DIR}/parallel)

# add cpp17
set(cpp17_dir ${CMAKE_SOURCE_DIR}/cpp17)
function(add_cpp17 filename)
add_executable(${filename} ${cpp17_dir}/${filename}.cpp)
add_test(NAME ${filename} COMMAND ${filename})
endfunction()

add_cpp17(struct01)
add_cpp17(struct02)
add_executable(inline_test ${cpp17_dir}/inline01.cpp ${cpp17_dir}/inline02.cpp)
add_test(NAME inline_test COMMAND inline_test)
