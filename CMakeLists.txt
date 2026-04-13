cmake_minimum_required(VERSION 3.22)
project(cgal_point_cloud_skeletonization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_executable(
  skeletonization
  skeletonization.cpp
  mesh_reconstruction.cpp
)

target_link_libraries(skeletonization PRIVATE CGAL::CGAL)
target_link_libraries(skeletonization PRIVATE Eigen3::Eigen)
target_compile_definitions(skeletonization PRIVATE CGAL_EIGEN3_ENABLED)

# CGAL packages used here rely on Eigen-based sparse solvers.
if(TARGET CGAL::Eigen3_support)
  target_link_libraries(skeletonization PRIVATE CGAL::Eigen3_support)
endif()
