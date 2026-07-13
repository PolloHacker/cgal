#include "io_visualization.h"
#include "pipeline_config.h"
#include "point_cloud_processing.h"

#include <cstdlib>
#include <iostream>
#include <vector>

using Point_set = mesh_reconstruction::Point_set;

/** \brief Point cloud preprocessing main function. */
int main(int argc, char *argv[]) {
#ifndef CGAL_LINKED_WITH_TBB
  std::cerr << "Warning: CGAL_LINKED_WITH_TBB is not defined. "
            << "TBB support is recommended for parallel processing.\n";
#endif

  Pipeline_options options;
  if (!parse_args(argc, argv, options)) {
    return EXIT_FAILURE;
  }

  if (!prepare_output_dir(options.output_dir)) {
    return EXIT_FAILURE;
  }

  const Output_paths output_paths = make_output_paths(options);

  Point_set points;
  if (!load_oriented_points(options.input_path, points, options)) {
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
          "1.2 Visualize preprocessed point cloud")) {
    return EXIT_FAILURE;
  }

  std::cout << "\nPreprocessing completed successfully. Output saved to: "
            << output_paths.preprocessed_points.string() << "\n";
  return EXIT_SUCCESS;
}
