set(JSON_SRC_DIR ${CMAKE_CURRENT_LIST_DIR})
include_directories(${ROOT_DIR}/3rd/json/include)

add_executable(json_test1 ${JSON_SRC_DIR}/json_test1.cpp)
add_test(NAME json_test1 COMMAND json_test1)