#include "skeleton_extraction.h"

#include "io_visualization.h"

#include <CGAL/boost/graph/split_graph_into_polylines.h>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

using Triangle_mesh = mesh_reconstruction::Triangle_mesh;
using mesh_vertex_descriptor =
    boost::graph_traits<Triangle_mesh>::vertex_descriptor;

using Skeleton_vertex = Skeleton::vertex_descriptor;
using Skeleton_edge = Skeleton::edge_descriptor;

namespace {

/** \brief Visitor used to serialize skeleton maximal polylines. */
struct Display_polylines {
  const Skeleton &skeleton;
  std::ofstream &out;
  bool is_first = true;

  // The visitor is called by CGAL::split_graph_into_polylines for each maximal
  // polyline in the skeleton. It streams each vertex coordinate on its own row,
  // with a blank line separating individual polylines.
  Display_polylines(const Skeleton &skeleton_ref, std::ofstream &out_ref)
      : skeleton(skeleton_ref), out(out_ref) {}

  void start_new_polyline() {
    if (!is_first) {
      out << "\n";
    }
    is_first = false;
  }

  void add_node(const Skeleton_vertex v) {
    out << skeleton[v].point << "\n";
  }

  void end_polyline() {}
};

/** \brief Writes maximal skeleton polylines as one polyline per row. */
bool write_skeleton_polylines(const std::filesystem::path &out_path,
                              const Skeleton &skeleton) {
  std::ofstream out(out_path.string());
  if (!out) {
    std::cerr << "Error: cannot open " << out_path.string()
              << " for writing.\n";
    return false;
  }

  Display_polylines display(skeleton, out);
  CGAL::split_graph_into_polylines(skeleton, display);
  return true;
}

/** \brief Writes the skeleton as a Wavefront OBJ file (vertices and lines). */
bool write_skeleton_edges(const std::filesystem::path &out_path,
                          const Skeleton &skeleton) {
  std::ofstream out(out_path.string());
  if (!out) {
    std::cerr << "Error: cannot open " << out_path.string()
              << " for writing.\n";
    return false;
  }

  out << "# Wavefront OBJ skeleton edges\n";
  std::map<Skeleton_vertex, int> vertex_indices;
  int index = 1;
  for (const Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
    vertex_indices[v] = index++;
    out << "v " << skeleton[v].point << "\n";
  }

  for (const Skeleton_edge edge : CGAL::make_range(edges(skeleton))) {
    const Skeleton_vertex source_v = source(edge, skeleton);
    const Skeleton_vertex target_v = target(edge, skeleton);
    out << "l " << vertex_indices[source_v] << " " << vertex_indices[target_v] << "\n";
  }
  return true;
}

/** \brief Writes vertex-to-skeleton correspondence as lines in a Wavefront OBJ file. */
bool write_correspondence(const std::filesystem::path &out_path,
                          const Skeleton &skeleton,
                          const Triangle_mesh &mesh) {
  std::ofstream out(out_path.string());
  if (!out) {
    std::cerr << "Error: cannot open " << out_path.string()
              << " for writing.\n";
    return false;
  }

  out << "# Wavefront OBJ vertex-to-skeleton correspondence\n";
  int vertex_index = 1;
  for (const Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
    for (const mesh_vertex_descriptor vd : skeleton[v].vertices) {
      out << "v " << skeleton[v].point << "\n";
      out << "v " << get(CGAL::vertex_point, mesh, vd) << "\n";
      out << "l " << vertex_index << " " << (vertex_index + 1) << "\n";
      vertex_index += 2;
    }
  }
  return true;
}

} // namespace

std::unique_ptr<Skeletonization> MeanCurvatureFlowSkeletonizer::create_mcf(const mesh_reconstruction::Triangle_mesh &mesh) const {
  auto mcf = std::make_unique<Skeletonization>(mesh);

  mcf->set_max_triangle_angle(params_.max_triangle_angle);
  if (params_.min_edge_length > 0.0) {
    mcf->set_min_edge_length(params_.min_edge_length);
  }
  mcf->set_max_iterations(params_.max_iterations);
  mcf->set_area_variation_factor(params_.area_variation_factor);
  mcf->set_quality_speed_tradeoff(params_.quality_speed_tradeoff);
  mcf->set_is_medially_centered(params_.is_medially_centered);
  mcf->set_medially_centered_speed_tradeoff(params_.medially_centered_speed_tradeoff);

  return mcf;
}

bool MeanCurvatureFlowSkeletonizer::skeletonize(Triangle_mesh &mesh, Skeleton &skeleton) {
  log_stage("3. Mean curvature flow skeletonization");

  auto mcf = create_mcf(mesh);
  (*mcf)(skeleton);

  std::cout << "Skeleton vertices: " << boost::num_vertices(skeleton) << "\n";
  std::cout << "Skeleton edges: " << boost::num_edges(skeleton) << "\n";

  return boost::num_vertices(skeleton) > 0 && boost::num_edges(skeleton) > 0;
}

bool skeletonize(Triangle_mesh &mesh, Skeleton &skeleton) {
  MeanCurvatureFlowSkeletonizer default_skeletonizer;
  return default_skeletonizer.skeletonize(mesh, skeleton);
}

bool write_skeleton_outputs(const std::string &output_prefix, const Skeleton &skeleton,
                            const Triangle_mesh &mesh) {
  log_stage("3.1 Export skeleton artifacts");

  std::filesystem::path skeleton_polylines = output_prefix + "_skeleton.poly";
  std::filesystem::path skeleton_edges = output_prefix + "_skeleton_edges.obj";
  std::filesystem::path correspondence = output_prefix + "_correspondence.obj";

  // Clean up any old .txt output files that might remain from previous runs with this prefix.
  std::filesystem::remove(output_prefix + "_skeleton.polylines.txt");
  std::filesystem::remove(output_prefix + "_skeleton_edges.txt");
  std::filesystem::remove(output_prefix + "_correspondence.polylines.txt");

  if (!write_skeleton_polylines(skeleton_polylines, skeleton)) {
    return false;
  }

  if (!write_skeleton_edges(skeleton_edges, skeleton)) {
    return false;
  }

  if (!write_correspondence(correspondence, skeleton, mesh)) {
    return false;
  }

  std::cout << "Saved: " << skeleton_polylines.string() << "\n";
  std::cout << "Saved: " << skeleton_edges.string() << "\n";
  std::cout << "Saved: " << correspondence.string() << "\n";
  return true;
}
