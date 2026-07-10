#include "point_cloud_processing.h"

#include "io_visualization.h"
#include "validation.h"
#include "ply_io.h"
#include "attribute_transfer.h"

#include <CGAL/pca_estimate_normals.h>
#include <CGAL/jet_smooth_point_set.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/remove_outliers.h>
#include <CGAL/wlop_simplify_and_regularize_point_set.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

using Point = mesh_reconstruction::Point;
using Vector = mesh_reconstruction::Vector;
using Point_set = mesh_reconstruction::Point_set;

namespace {

/**
 * \brief Smooths point set coordinates locally using a parametric jet surface approximation.
 * \param points Point set to smooth (in-place).
 * \param requested_neighbors Neighborhood size (k) used for fitting.
 * \return true if smoothing succeeded.
 */
bool smooth_points(Point_set &points, const int requested_neighbors) {
  if (!validate_point_set(points, "smoothing input", false)) {
    return false;
  }

  const std::size_t max_neighbors = points.size() - 1;
  const unsigned int neighbors =
      static_cast<unsigned int>(std::min<std::size_t>(
          static_cast<std::size_t>(requested_neighbors), max_neighbors));

  if (neighbors < 2) {
    std::cerr << "Error: smoothing requires at least 2 neighbors.\n";
    return false;
  }

  CGAL::jet_smooth_point_set<CGAL::Parallel_tag>(points, neighbors);

  std::cout << "Smoothing neighbors: " << neighbors << "\n";
  return validate_point_set(points, "post-smoothing point set", false);
}

/**
 * \brief Simplifies the point cloud using Weighted Locally Optimal Projection (WLOP)
 *        and maps attributes back to the new regularized positions.
 * \param points Point set to downsample (in-place).
 * \param options Options structure containing iterations, radius, and retention percent.
 * \return true if WLOP and attribute transfer succeed.
 */
bool apply_wlop_downsampling(Point_set &points,
                             const Pipeline_options &options) {
  if (!validate_point_set(points, "WLOP input", false)) {
    return false;
  }

  const std::size_t input_count = points.size();
  std::vector<Point> downsampled_points;
  downsampled_points.reserve(
      std::max<std::size_t>(3, static_cast<std::size_t>(std::ceil(
                                   static_cast<double>(input_count) *
                                   (options.wlop_retain_percent / 100.0)))));

  const double wlop_radius =
      (options.wlop_neighbor_radius <= 0.0)
          ? -1.0
          : options.wlop_neighbor_radius *
                compute_average_spacing(points, options.outlier_neighbors);

  CGAL::wlop_simplify_and_regularize_point_set<CGAL::Parallel_tag>(
      points, std::back_inserter(downsampled_points),
      CGAL::parameters::point_map(points.point_map())
          .select_percentage(options.wlop_retain_percent)
          .neighbor_radius(wlop_radius)
          .number_of_iterations(options.wlop_iterations)
          .require_uniform_sampling(options.wlop_require_uniform_sampling));

  if (downsampled_points.size() < 3) {
    std::cerr << "Error: WLOP produced too few points ("
              << downsampled_points.size()
              << "). Increase --wlop-retain-percent.\n";
    return false;
  }

  // Set up target Point Set with only the regularized positions
  Point_set final_points;
  for (const auto &p : downsampled_points) {
    final_points.insert(p);
  }

  // Transfer original attributes to new regularized positions
  std::cout << "WLOP complete. Building 1-NN KD-tree to transfer attributes...\n";
  if (!mesh_reconstruction::transfer_attributes(points, final_points)) {
      return false;
  }

  points = std::move(final_points);

  std::cout << "WLOP retain percent: " << options.wlop_retain_percent << "\n";
  std::cout << "WLOP neighbor radius: " << options.wlop_neighbor_radius
            << (options.wlop_neighbor_radius <= 0.0 ? " (auto)" : "") << "\n";
  std::cout << "WLOP iterations: " << options.wlop_iterations << "\n";
  std::cout << "WLOP uniform sampling: "
            << (options.wlop_require_uniform_sampling ? "enabled" : "disabled")
            << "\n";
  std::cout << "WLOP points: " << input_count << " -> " << points.size()
            << "\n";

  return true;
}

} // namespace

bool load_oriented_points(const std::string &input_path,
                          Point_set &points,
                          const Pipeline_options &options) {
  if (options.enable_spatial_subsampling) {
    log_stage("1. Load point cloud with Spatial Subsampling");
  } else {
    log_stage("1. Load point cloud + normals (PLY)");
  }
  
  if (!mesh_reconstruction::load_ply(input_path, points, options.enable_spatial_subsampling, options.spatial_subsample_distance)) {
    return false;
  }

  if (!validate_point_set(points, "loaded point set", false)) {
    return false;
  }

  std::cout << "Loaded points: " << points.size() << "\n";
  return true;
}

bool preprocess_points(Point_set &points, const Pipeline_options &options,
                       double &average_spacing) {
  if (!validate_point_set(points, "loaded point set", false)) {
    return false;
  }

  if (options.outlier_percent > 0.0) {
    log_stage("1.2 Outlier removal");
    average_spacing = compute_average_spacing(points, options.outlier_neighbors);

    auto rout_it = CGAL::remove_outliers<CGAL::Parallel_tag>(
        points, 
        options.outlier_neighbors, 
        points.parameters().threshold_percent(options.outlier_percent)
                           .threshold_distance(average_spacing)
    );
    const std::size_t before = points.size();
    points.remove(rout_it, points.end());
    points.collect_garbage();
    std::cout << "Outlier removal: " << (before - points.size())
              << " points removed.\n";
  } else {
    std::cout << "Outlier removal: disabled.\n";
  }

  if (!validate_point_set(points, "post-outlier point set", false)) {
    return false;
  }

  if (options.enable_wlop) {
    log_stage("1.3 WLOP downsampling");
    if (options.wlop_retain_percent < 10.0) {
      std::cout << "Warning: WLOP retain percent is very low and may "
                   "destabilize Poisson reconstruction.\n";
    }
    if (!apply_wlop_downsampling(points, options)) {
      return false;
    }
  } else {
    std::cout << "WLOP downsampling: disabled.\n";
  }

  if (!validate_point_set(points, "post-WLOP point set", false)) {
    return false;
  }

  if (options.enable_smoothing) {
    log_stage("1.4 Jet smoothing");
    if (!smooth_points(points, options.smoothing_neighbors)) {
      return false;
    }
  } else {
    std::cout << "Smoothing: disabled.\n";
  }

  if (!validate_point_set(points, "preprocessed point set", false)) {
    return false;
  }

  log_stage("1.6 Average spacing");
  average_spacing = compute_average_spacing(points, options.outlier_neighbors);

  if (!validate_average_spacing(average_spacing, "average spacing")) {
    return false;
  }

  std::cout << "Average spacing (k=" << options.outlier_neighbors << "): " << average_spacing << "\n";
  return true;
}
