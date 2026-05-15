cmake_minimum_required(VERSION 3.22)
project(cgal_point_cloud_skeletonization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If the vcpkg toolchain is not configured explicitly, try common Windows vcpkg
# locations so local VS Code/CMake Tools configure still works.
set(_vcpkg_prefix_hints)
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(_vcpkg_roots)
  if(DEFINED ENV{VCPKG_ROOT} AND NOT "$ENV{VCPKG_ROOT}" STREQUAL "")
    list(APPEND _vcpkg_roots "$ENV{VCPKG_ROOT}")
  endif()
  if(DEFINED ENV{USERPROFILE} AND NOT "$ENV{USERPROFILE}" STREQUAL "")
    list(APPEND _vcpkg_roots "$ENV{USERPROFILE}/vcpkg")
  endif()

  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_default_triplet x64-windows)
    set(_arch_triplet_prefix x64-)
  else()
    set(_default_triplet x86-windows)
    set(_arch_triplet_prefix x86-)
  endif()

  set(_vcpkg_triplets "${_default_triplet}")

  set(_env_triplets)
  if(DEFINED ENV{VCPKG_TARGET_TRIPLET} AND NOT "$ENV{VCPKG_TARGET_TRIPLET}" STREQUAL "")
    list(APPEND _env_triplets "$ENV{VCPKG_TARGET_TRIPLET}")
  endif()
  if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT "$ENV{VCPKG_DEFAULT_TRIPLET}" STREQUAL "")
    list(APPEND _env_triplets "$ENV{VCPKG_DEFAULT_TRIPLET}")
  endif()

  foreach(_env_triplet IN LISTS _env_triplets)
    if(_env_triplet MATCHES "^${_arch_triplet_prefix}")
      list(APPEND _vcpkg_triplets "${_env_triplet}")
    else()
      message(STATUS
        "Ignoring ${_env_triplet}: does not match generator architecture; expected prefix ${_arch_triplet_prefix}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _vcpkg_roots)
  list(REMOVE_DUPLICATES _vcpkg_triplets)

  foreach(_root IN LISTS _vcpkg_roots)
    foreach(_triplet IN LISTS _vcpkg_triplets)
      set(_vcpkg_prefix "${_root}/installed/${_triplet}")
      if(EXISTS "${_vcpkg_prefix}")
        list(APPEND _vcpkg_prefix_hints "${_vcpkg_prefix}")
      endif()
    endforeach()
  endforeach()

  if(_vcpkg_prefix_hints)
    list(APPEND CMAKE_PREFIX_PATH ${_vcpkg_prefix_hints})
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
  endif()
endif()

find_package(CGAL CONFIG QUIET HINTS ${_vcpkg_prefix_hints})
if(NOT CGAL_FOUND)
  if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR
      "CGAL not found in architecture-matching vcpkg prefixes. Use a triplet matching the "
      "generator architecture (x64 <-> x64-windows, Win32 <-> x86-windows), or configure with "
      "CMAKE_TOOLCHAIN_FILE. On Linux, install CGAL and make sure CGALConfig.cmake is discoverable.")
  endif()
  find_package(CGAL CONFIG REQUIRED)
endif()

find_package(Eigen3 CONFIG QUIET HINTS ${_vcpkg_prefix_hints})
if(NOT Eigen3_FOUND)
  if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR
      "Eigen3 not found in architecture-matching vcpkg prefixes. Install eigen3 for the active "
      "triplet or configure with CMAKE_TOOLCHAIN_FILE. On Linux, install Eigen3 and make sure "
      "Eigen3Config.cmake is discoverable.")
  endif()
  find_package(Eigen3 CONFIG REQUIRED)
endif()

find_package(TBB REQUIRED)

add_executable(
  skeletonization
  src/skeletonization.cpp
  src/pipeline_config.cpp
  src/point_cloud_processing.cpp
  src/io_visualization.cpp
  src/skeleton_extraction.cpp
  src/mesh_reconstruction.cpp
  src/validation.cpp
)

target_include_directories(skeletonization PRIVATE include)

target_link_libraries(skeletonization PRIVATE 
  CGAL::CGAL 
  Eigen3::Eigen
  TBB::tbb
)

target_compile_definitions(skeletonization PRIVATE 
  CGAL_EIGEN3_ENABLED 
  CGAL_LINKED_WITH_TBB 
)

# CGAL packages used here rely on Eigen-based sparse solvers.
if(TARGET CGAL::Eigen3_support)
  target_link_libraries(skeletonization PRIVATE CGAL::Eigen3_support)
endif()
