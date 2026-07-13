cmake_minimum_required(VERSION 3.20)
project(wnnc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# Core library: octree, treecode operators, solver, point-cloud I/O.
add_library(wnnc_core src/wnnc/point_cloud_io.cpp)
target_include_directories(wnnc_core PUBLIC src)
target_link_libraries(wnnc_core PUBLIC Threads::Threads)
if(NOT MSVC)
  # sqrt() need not set errno; this lets it compile to a bare instruction
  # inside the hot treecode kernels. (No effect on results.)
  target_compile_options(wnnc_core PUBLIC -fno-math-errno)
endif()

# Command-line tool (counterpart of the reference main_wnnc.py).
add_executable(wnnc src/main.cpp)
target_include_directories(wnnc PRIVATE third_party/CLI11)
target_link_libraries(wnnc PRIVATE wnnc_core)

enable_testing()
add_executable(wnnc_tests tests/treecode_tests.cpp)
target_link_libraries(wnnc_tests PRIVATE wnnc_core)
add_test(NAME treecode COMMAND wnnc_tests)
