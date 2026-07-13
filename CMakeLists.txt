cmake_minimum_required(VERSION 3.22)
project(cgal_reconstruction_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(TBB REQUIRED)

# WNNC Core library
add_library(wnnc_core 
  src/wnnc/cpp/src/wnnc/point_cloud_io.cpp
)
target_include_directories(wnnc_core PUBLIC src/wnnc/cpp/src)
target_link_libraries(wnnc_core PUBLIC Threads::Threads)
if(NOT MSVC)
  target_compile_options(wnnc_core PUBLIC -fno-math-errno)
endif()

# Original WNNC binary (standalone CLI)
add_executable(wnnc src/wnnc/cpp/src/main.cpp)
target_include_directories(wnnc PRIVATE src/wnnc/cpp/third_party/CLI11)
target_link_libraries(wnnc PRIVATE wnnc_core)

# Original Preprocessor and Skeletonization subdirectories (keep them compilable)
add_subdirectory(src/preprocess)

# Add the main unified pipeline target
add_executable(pipeline
  src/pipeline/pipeline_main.cpp
  src/pipeline/poisson_recon_wrapper.cpp
  src/preprocess/src/pipeline_config.cpp
  src/preprocess/src/point_cloud_processing.cpp
  src/preprocess/src/ply_io.cpp
  src/preprocess/src/attribute_transfer.cpp
  src/preprocess/src/io_visualization.cpp
  src/preprocess/src/mesh_reconstruction.cpp
  src/preprocess/src/validation.cpp
  src/preprocess/src/skeleton_extraction.cpp
)

target_include_directories(pipeline PRIVATE 
  src/preprocess/include
  src/wnnc/cpp/src
  src/wnnc/cpp/third_party/CLI11
  src/AdaptiveSolvers/Src
)

target_link_libraries(pipeline PRIVATE 
  wnnc_core
  CGAL::CGAL 
  Eigen3::Eigen
  TBB::tbb
  OpenMP::OpenMP_CXX
)

target_compile_definitions(pipeline PRIVATE 
  CGAL_EIGEN3_ENABLED 
  CGAL_LINKED_WITH_TBB 
  USE_DOUBLE
)

if(TARGET CGAL::Eigen3_support)
  target_link_libraries(pipeline PRIVATE CGAL::Eigen3_support)
endif()

# TDD test target
enable_testing()
add_executable(pipeline_tests
  src/pipeline/tests/pipeline_tests.cpp
  src/pipeline/poisson_recon_wrapper.cpp
)

target_include_directories(pipeline_tests PRIVATE
  src/preprocess/include
  src/pipeline
  src/AdaptiveSolvers/Src
)

target_link_libraries(pipeline_tests PRIVATE
  CGAL::CGAL
  Eigen3::Eigen
  TBB::tbb
  OpenMP::OpenMP_CXX
  Threads::Threads
)

target_compile_definitions(pipeline_tests PRIVATE
  USE_DOUBLE
)

if(TARGET CGAL::Eigen3_support)
  target_link_libraries(pipeline_tests PRIVATE CGAL::Eigen3_support)
endif()

add_test(NAME pipeline_tests COMMAND pipeline_tests)
