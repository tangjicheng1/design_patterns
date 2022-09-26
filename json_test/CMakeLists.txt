enable_testing()

set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(json_test1 ${SRC_DIR}/json_test1.cpp)
add_test(NAME json_test1 COMMAND json_test1)