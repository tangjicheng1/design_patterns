cmake_minimum_required(VERSION 3.0)
project(design_patterns LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)

option(build_cpp_lang "Build cpp_lang dir" ON)
option(build_patterns "Build patterns dir" ON)
option(build_parallel "Build cpp_lang dir" ON)
option(build_not_single "Build cpp_lang dir" ON)
option(build_json_test "Build json test dir" ON)

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
  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(${cur_exe} pthread atomic)
  endif(CMAKE_SYSTEM_NAME MATCHES "Linux")
  add_test(NAME ${cur_exe} COMMAND ${cur_exe})
endforeach()
endfunction()

if(build_cpp_lang)
  add_each(${ROOT_DIR}/cpp_lang)
endif(build_cpp_lang)

if(build_parallel)
  add_each(${ROOT_DIR}/parallel)
endif(build_parallel)

if(build_patterns)
  add_each(${ROOT_DIR}/patterns)
endif(build_patterns)

# add not_single_dir
set(not_single_dir ${CMAKE_SOURCE_DIR}/not_single)
if(build_not_single)
  add_executable(inline_test ${not_single_dir}/inline01.cpp ${not_single_dir}/inline02.cpp)
  add_executable(fstream_test ${not_single_dir}/fstream_test.cpp)
  add_executable(extern_test ${not_single_dir}/extern_test_01.cpp ${not_single_dir}/extern_test_02.cpp)
endif(build_not_single)

# json test
if(build_json_test)
  include_directories(3rd/json/include)
  include(json_test/CMakeLists.txt)
endif(build_json_test)