cmake_minimum_required(VERSION 3.0)
project(design_patterns LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)

enable_testing()

# add patterns
set(patterns_dir ${CMAKE_SOURCE_DIR}/patterns)
function(add_patterns filename)
add_executable(${filename} ${patterns_dir}/${filename}.cpp)
add_test(NAME ${filename} COMMAND ${filename})
endfunction()

# add cpp_lang
set(cpp_lang_dir ${CMAKE_SOURCE_DIR}/cpp_lang)
function(add_langs filename)
add_executable(${filename} ${cpp_lang_dir}/${filename}.cpp)
add_test(NAME ${filename} COMMAND ${filename})
endfunction()

# add cpp11
set(cpp11_dir ${CMAKE_SOURCE_DIR}/cpp11)
function(add_cpp11 filename)
add_executable(${filename} ${cpp11_dir}/${filename}.cpp)
add_test(NAME ${filename} COMMAND ${filename})
endfunction()

# add cpp17
set(cpp17_dir ${CMAKE_SOURCE_DIR}/cpp17)
function(add_cpp17 filename)
add_executable(${filename} ${cpp17_dir}/${filename}.cpp)
add_test(NAME ${filename} COMMAND ${filename})
endfunction()

add_patterns(singleton)
add_patterns(factory)
add_patterns(pimpl)
add_patterns(facade)
add_patterns(templ)
add_patterns(strategy)
add_patterns(table_driven)
add_patterns(chain_of_responsibility)
add_patterns(crtp)
add_patterns(adl)
add_patterns(builder)
add_patterns(bridge)

add_langs(cpp_lang01)
add_langs(cpp_lang02)
add_langs(cpp_lang03)
add_langs(cpp_lang04)
add_langs(cpp_lang05)
# include_directories(/Users/tangjicheng/spdlog/include)
# add_langs(printf_cout)
add_langs(global_var1)
add_langs(array_ref_test)
add_langs(vector)

add_cpp11(tuple01)
add_cpp11(integral_constant01)
add_cpp11(initializer_list1)
add_cpp11(array1)

add_cpp17(struct01)
add_cpp17(struct02)
add_executable(inline_test ${cpp17_dir}/inline01.cpp ${cpp17_dir}/inline02.cpp)
add_test(NAME inline_test COMMAND inline_test)
