#include "point_cloud_processing.h"

#include "io_visualization.h"
#include "validation.h"
#include "ply_io.h"
#include "attribute_transfer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

using Point = mesh_reconstruction::Point;
using Vector = mesh_reconstruction::Vector;
using Point_set = mesh_reconstruction::Point_set;


bool load_oriented_points(const std::string &input_path,
                          Point_set &points,
                          const Pipeline_options &options) {
  if (options.enable_spatial_subsampling) {
    log_stage("1. Load point cloud with Spatial Subsampling");
  } else {
    log_stage("1. Load point cloud + normals (PLY)");
  }
  
  if (!mesh_reconstruction::load_ply(input_path, points, options.enable_spatial_subsampling, options.spatial_subsample_distance, options.enable_cuda)) {
    return false;
  }

  if (!validate_point_set(points, "loaded point set", false)) {
    return false;
  }

  std::cout << "Loaded points: " << points.size() << "\n";
  return true;
}