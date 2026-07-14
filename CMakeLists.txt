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

option(WNNC_USE_CUDA "Use CUDA acceleration for WNNC solver" ON)

if(WNNC_USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Let CMake auto-detect and target the host GPU compute architecture
    set(CMAKE_CUDA_ARCHITECTURES native)
    message(STATUS "CUDA compilation enabled.")
  else()
    message(STATUS "CUDA compiler not found. Falling back to CPU-only build.")
    set(WNNC_USE_CUDA OFF)
  endif()
endif()

# Common sources for WNNC
set(WNNC_CORE_SOURCES
  src/wnnc/cpp/src/wnnc/point_cloud_io.cpp
  src/wnnc/python/ext/wn_treecode/wn_treecode_cpu/wn_treecode_cpu_treeutils.cpp
)

set(WNNC_CORE_INCLUDES
  src/wnnc/cpp/src
  src/wnnc/python/ext/wn_treecode/wn_treecode_cpu
)

if(WNNC_USE_CUDA)
  list(APPEND WNNC_CORE_SOURCES
    src/wnnc/python/ext/wn_treecode/wn_treecode_cuda/wn_treecode_cuda_kernels.cu
    src/pipeline/winding_number_operator_cuda.cu
  )
  list(APPEND WNNC_CORE_INCLUDES
    src/wnnc/python/ext/wn_treecode/wn_treecode_cuda
  )
else()
  list(APPEND WNNC_CORE_SOURCES
    src/wnnc/python/ext/wn_treecode/wn_treecode_cpu/wn_treecode_cpu_kernels.cpp
    src/pipeline/winding_number_operator_cpu.cpp
  )
endif()

add_library(wnnc_core ${WNNC_CORE_SOURCES})

target_include_directories(wnnc_core PUBLIC ${WNNC_CORE_INCLUDES})

target_link_libraries(wnnc_core PUBLIC 
  Threads::Threads
  OpenMP::OpenMP_CXX
)

if(WNNC_USE_CUDA)
  target_link_libraries(wnnc_core PUBLIC CUDA::cudart)
  target_compile_definitions(wnnc_core PUBLIC WNNC_USE_CUDA)
endif()

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
)

target_include_directories(pipeline PRIVATE 
  src/preprocess/include
  src/wnnc/cpp/src
  src/wnnc/cpp/third_party/CLI11
  src/AdaptiveSolvers/Src
)

target_link_libraries(pipeline PRIVATE 
  wnnc_core
  preprocess_core
  TBB::tbb
)

target_compile_definitions(pipeline PRIVATE 
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
  src/wnnc/cpp/src
)

target_link_libraries(pipeline_tests PRIVATE
  wnnc_core
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
