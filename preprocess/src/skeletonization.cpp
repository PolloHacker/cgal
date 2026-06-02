#include "io_visualization.h"
#include "mesh_reconstruction.h"
#include "pipeline_config.h"
#include "point_cloud_processing.h"
#include "skeleton_extraction.h"

#include <cstdlib>
#include <iostream>
#include <vector>

using Pwn = mesh_reconstruction::Pwn;
using Triangle_mesh = mesh_reconstruction::Triangle_mesh;

/** \brief point-cloud to skeleton pipeline. */
int main(int argc, char *argv[]) {
#ifndef CGAL_EIGEN3_ENABLED
  std::cerr << "Error: CGAL_EIGEN3_ENABLED is not defined. "
            << "Install Eigen and configure CGAL with Eigen support.\n";
  return EXIT_FAILURE;
#endif

#ifndef CGAL_LINKED_WITH_TBB
  std::cerr << "Error: CGAL_LINKED_WITH_TBB is not defined. "
            << "Install TBB and configure CGAL with TBB support for parallel "
               "processing.\n";
  return EXIT_FAILURE;
#endif

  Pipeline_options options;
  if (!parse_args(argc, argv, options)) {
    return EXIT_FAILURE;
  }

  if (!prepare_output_dir(options.output_dir)) {
    return EXIT_FAILURE;
  }

  const Output_paths output_paths = make_output_paths(options);

  std::vector<Pwn> points;
  if (!load_oriented_points(options.input_path, points)) {
    return EXIT_FAILURE;
  }

  if (!write_point_stage_visualization(output_paths.raw_points, points,
                                       "1.1 Visualize raw input point cloud")) {
    return EXIT_FAILURE;
  }

  double average_spacing = 0.0;
  if (!preprocess_points(points, options, average_spacing)) {
    return EXIT_FAILURE;
  }

  if (!write_point_stage_visualization(
          output_paths.preprocessed_points, points,
          "1.7 Visualize preprocessed point cloud")) {
    return EXIT_FAILURE;
  }

  /*
  Triangle_mesh mesh;

  log_stage("2. Poisson reconstruction");
  if (!mesh_reconstruction::reconstruct_mesh_poisson(
          points, average_spacing, mesh, options.sm_angle, options.sm_radius,
          options.sm_distance)) {
    return EXIT_FAILURE;
  }

  if (!write_mesh_stage_visualization(
          output_paths.raw_reconstructed_mesh, mesh,
          "2.1 Visualize raw Poisson reconstruction")) {
    return EXIT_FAILURE;
  }

  log_stage("2.2 Mesh normalization and validation");
  if (!mesh_reconstruction::normalize_mesh_for_skeletonization(
          mesh, options.keep_largest_component)) {
    return EXIT_FAILURE;
  }

  if (!write_mesh_stage_visualization(
          output_paths.reconstructed_mesh, mesh,
          "2.3 Export normalized reconstructed mesh")) {
    return EXIT_FAILURE;
  }

  Skeleton skeleton;
  if (!skeletonize(mesh, skeleton)) {
    std::cerr << "Error: skeleton extraction produced an empty skeleton.\n";
    return EXIT_FAILURE;
  }

  if (!write_skeleton_outputs(output_paths, skeleton, mesh)) {
    return EXIT_FAILURE;
  }
  */

  std::cout << "\nPipeline completed successfully.\n";
  return EXIT_SUCCESS;
}
