cmake_minimum_required(VERSION 3.22)
project(cgal_reconstruction_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Dependencies
# ─────────────────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)
find_package(OpenMP  REQUIRED)
find_package(CGAL    CONFIG REQUIRED)
find_package(Eigen3  CONFIG REQUIRED)
find_package(TBB     REQUIRED)

option(USE_CUDA "Use CUDA acceleration for WNNC solver and Parallel Decimator" ON)

if(USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES native)
    message(STATUS "CUDA compilation enabled.")
  else()
    message(STATUS "CUDA compiler not found. Falling back to CPU-only build.")
    set(USE_CUDA OFF)
  endif()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveSolvers — header-only Poisson/FEM template library (Kazhdan)
# ─────────────────────────────────────────────────────────────────────────────
add_library(adaptive_solvers INTERFACE)
target_include_directories(adaptive_solvers INTERFACE src/AdaptiveSolvers/Src)

# ─────────────────────────────────────────────────────────────────────────────
# preprocess_core — CGAL preprocessing, decimation, skeleton extraction
# ─────────────────────────────────────────────────────────────────────────────
add_library(preprocess_core STATIC
  src/preprocess/src/pipeline_config.cpp
  src/preprocess/src/point_cloud_processing.cpp
  src/preprocess/src/ply_io.cpp
  src/preprocess/src/attribute_transfer.cpp
  src/preprocess/src/io_visualization.cpp
  src/preprocess/src/mesh_reconstruction.cpp
  src/preprocess/src/validation.cpp
  src/preprocess/src/skeleton_extraction.cpp
  src/preprocess/src/parallel_decimator.cpp
  src/preprocess/src/tbb_cpu_decimator.cpp
  src/preprocess/src/mmap_ply_reader.cpp
  src/preprocess/src/cuda_decimator.cu
)

target_include_directories(preprocess_core PUBLIC src/preprocess/include)

target_link_libraries(preprocess_core PUBLIC
  CGAL::CGAL
  Eigen3::Eigen
  TBB::tbb
)

target_compile_definitions(preprocess_core PUBLIC
  CGAL_EIGEN3_ENABLED
  CGAL_LINKED_WITH_TBB
)

if(USE_CUDA)
  set_source_files_properties(src/preprocess/src/cuda_decimator.cu PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(preprocess_core PUBLIC USE_CUDA)
  target_link_libraries(preprocess_core PUBLIC CUDA::cudart)
else()
  set_source_files_properties(src/preprocess/src/cuda_decimator.cu PROPERTIES LANGUAGE CXX)
endif()

if(TARGET CGAL::Eigen3_support)
  target_link_libraries(preprocess_core PUBLIC CGAL::Eigen3_support)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# wnnc_core — winding number normal estimation (CPU or CUDA)
# ─────────────────────────────────────────────────────────────────────────────
set(WNNC_CORE_SOURCES
  src/wnnc/cpp/src/wnnc/point_cloud_io.cpp
  src/wnnc/python/ext/wn_treecode/wn_treecode_cpu/wn_treecode_cpu_treeutils.cpp
)

set(WNNC_CORE_INCLUDES
  src/wnnc/cpp/src
  src/wnnc/python/ext/wn_treecode/wn_treecode_cpu
)

if(USE_CUDA)
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

if(USE_CUDA)
  set_target_properties(wnnc_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_link_libraries(wnnc_core PUBLIC CUDA::cudart)
  target_compile_definitions(wnnc_core PUBLIC WNNC_USE_CUDA USE_CUDA)
endif()

if(NOT MSVC)
  target_compile_options(wnnc_core PUBLIC -fno-math-errno)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# pipeline — main unified executable
# ─────────────────────────────────────────────────────────────────────────────
add_executable(pipeline
  src/pipeline/pipeline_main.cpp
  src/pipeline/poisson_recon_wrapper.cpp
)

target_include_directories(pipeline PRIVATE
  src/wnnc/cpp/third_party/CLI11
)

target_link_libraries(pipeline PRIVATE
  wnnc_core
  preprocess_core
  adaptive_solvers
)

target_compile_definitions(pipeline PRIVATE USE_DOUBLE)

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
enable_testing()

# Pipeline integration tests
add_executable(pipeline_tests
  src/pipeline/tests/pipeline_tests.cpp
  src/pipeline/poisson_recon_wrapper.cpp
)

target_include_directories(pipeline_tests PRIVATE
  src/pipeline
)

target_link_libraries(pipeline_tests PRIVATE
  wnnc_core
  preprocess_core
  adaptive_solvers
)

target_compile_definitions(pipeline_tests PRIVATE USE_DOUBLE)

add_test(NAME pipeline_tests COMMAND pipeline_tests)

# Preprocess unit tests
add_executable(preprocess_tests
  src/preprocess/test/test_main.cpp
  src/preprocess/test/test_ply_io.cpp
  src/preprocess/test/test_attribute_transfer.cpp
  src/preprocess/test/test_validation.cpp
  src/preprocess/test/test_mesh_reconstruction.cpp
  src/preprocess/test/test_skeletonization.cpp
  src/preprocess/test/test_parallel_decimator.cpp
)

target_link_libraries(preprocess_tests PRIVATE preprocess_core)

add_test(NAME preprocess_tests COMMAND preprocess_tests)
