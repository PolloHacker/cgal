#include "io_visualization.h"
#include "mesh_reconstruction.h"
#include "skeleton_extraction.h"

#include <cstdlib>
#include <iostream>
#include <vector>

#if __has_include(<CGAL/IO/polygon_mesh_io.h>)
#include <CGAL/IO/polygon_mesh_io.h>
#elif __has_include(<CGAL/boost/graph/IO/polygon_mesh_io.h>)
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>
#else
#error "polygon_mesh_io header not found in this CGAL installation"
#endif

#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#if __has_include(<CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_stop_predicate.h>)
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_stop_predicate.h>
using Stop_predicate = CGAL::Surface_mesh_simplification::Edge_count_stop_predicate<mesh_reconstruction::Triangle_mesh>;
#elif __has_include(<CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_stop_predicate.h>)
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_stop_predicate.h>
using Stop_predicate = CGAL::Surface_mesh_simplification::Count_stop_predicate<mesh_reconstruction::Triangle_mesh>;
#else
#error "CGAL stop predicate header not found"
#endif

#include <CLI11.hpp>

using Triangle_mesh = mesh_reconstruction::Triangle_mesh;

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

  CLI::App app{"Standalone Mean Curvature Flow skeletonization"};

  std::string input_mesh_path;
  std::string output_prefix;
  SkeletonizationParameters params;
  bool no_medially_centered = false;

  app.add_option("input_mesh", input_mesh_path, "Input mesh file (.ply)")->required()->check(CLI::ExistingFile);
  app.add_option("output_prefix", output_prefix, "Prefix for output skeleton artifacts")->required();

  app.add_option("--max-triangle-angle", params.max_triangle_angle, "Max triangle angle threshold for face splitting")->capture_default_str();
  app.add_option("--min-edge-length", params.min_edge_length, "Min edge length for collapse (0.0 for auto-calculation)")->capture_default_str();
  app.add_option("--max-iterations", params.max_iterations, "Maximum number of iterations")->capture_default_str();
  app.add_option("--area-variation-factor", params.area_variation_factor, "Area variation factor for convergence")->capture_default_str();
  app.add_option("--quality-speed-tradeoff", params.quality_speed_tradeoff, "Quality/speed tradeoff parameter")->capture_default_str();
  app.add_flag("--no-medially-centered", no_medially_centered, "Disable medially centered attraction (default: active)");
  app.add_option("--medially-centered-speed-tradeoff", params.medially_centered_speed_tradeoff, "Medially centered speed tradeoff parameter")->capture_default_str();

  CLI11_PARSE(app, argc, argv);

  params.is_medially_centered = !no_medially_centered;

  std::cout << "Loading mesh from: " << input_mesh_path << "\n";
  Triangle_mesh mesh;
  if (!CGAL::IO::read_polygon_mesh(input_mesh_path, mesh)) {
    std::cerr << "Error: cannot read mesh from " << input_mesh_path << "\n";
    return EXIT_FAILURE;
  }

  std::cout << "Mesh loaded: " << num_vertices(mesh) << " vertices, "
            << num_faces(mesh) << " faces.\n";

  // Normalize mesh for skeletonization (triangulate, keep largest component, fill holes, stitch)
  // This guarantees that MCF skeletonization preconditions are met.
  std::cout << "Normalizing mesh...\n";
  if (!mesh_reconstruction::normalize_mesh_for_skeletonization(mesh, true)) {
    std::cerr << "Error: mesh normalization failed. Cannot proceed with skeletonization.\n";
    return EXIT_FAILURE;
  }

  // Decimate the mesh down to roughly 50,000 faces (approx 75,000 edges) to speed up skeletonization.
  std::size_t num_faces_before = num_faces(mesh);
  if (num_faces_before > 50000) {
    std::cout << "Decimating mesh from " << num_faces_before << " faces to ~50000 faces...\n";
    std::size_t target_edges = 75000;
    Stop_predicate stop(target_edges);
    int r = CGAL::Surface_mesh_simplification::edge_collapse(mesh, stop);
    std::cout << "Decimation finished. Removed " << r << " edges. Current mesh faces: " << num_faces(mesh) << "\n";

    // Re-normalize to ensure the decimated mesh is still perfectly closed and triangulated.
    std::cout << "Re-normalizing decimated mesh...\n";
    if (!mesh_reconstruction::normalize_mesh_for_skeletonization(mesh, true)) {
      std::cerr << "Error: post-decimation mesh normalization failed. Cannot proceed with skeletonization.\n";
      return EXIT_FAILURE;
    }
  }

  MeanCurvatureFlowSkeletonizer skeletonizer(params);
  Skeleton skeleton;
  if (!skeletonizer.skeletonize(mesh, skeleton)) {
    std::cerr << "Error: skeleton extraction produced an empty skeleton.\n";
    return EXIT_FAILURE;
  }

  if (!write_skeleton_outputs(output_prefix, skeleton, mesh)) {
    return EXIT_FAILURE;
  }

  std::cout << "\nSkeletonization completed successfully.\n";
  return EXIT_SUCCESS;
}
