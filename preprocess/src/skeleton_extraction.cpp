#include "skeleton_extraction.h"

#include "io_visualization.h"

#include <CGAL/boost/graph/split_graph_into_polylines.h>

#include <fstream>
#include <iostream>
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
  int polyline_size = 0;
  std::stringstream sstr;

  // The visitor is called by CGAL::split_graph_into_polylines for each maximal
  // polyline in the skeleton. It accumulates the points of the current polyline
  // and writes them to the output stream when the polyline ends.
  Display_polylines(const Skeleton &skeleton_ref, std::ofstream &out_ref)
      : skeleton(skeleton_ref), out(out_ref) {}

  void start_new_polyline() {
    polyline_size = 0;
    sstr.str("");
    sstr.clear();
  }

  void add_node(const Skeleton_vertex v) {
    ++polyline_size;
    sstr << " " << skeleton[v].point;
  }

  void end_polyline() { out << polyline_size << sstr.str() << "\n"; }
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

/** \brief Writes the skeleton as a plain edge list (segment endpoints).
 *
 * The plain edge list is good for applications that only need the skeleton
 * connectivity and geometry, without the maximal polyline structure.
 */
bool write_skeleton_edges(const std::filesystem::path &out_path,
                          const Skeleton &skeleton) {
  std::ofstream out(out_path.string());
  if (!out) {
    std::cerr << "Error: cannot open " << out_path.string()
              << " for writing.\n";
    return false;
  }

  out << "# x1 y1 z1 x2 y2 z2\n";
  for (const Skeleton_edge edge : CGAL::make_range(edges(skeleton))) {
    const Skeleton_vertex source_v = source(edge, skeleton);
    const Skeleton_vertex target_v = target(edge, skeleton);
    out << skeleton[source_v].point << " " << skeleton[target_v].point << "\n";
  }
  return true;
}

/** \brief Writes vertex-to-skeleton correspondence as line segments. */
bool write_correspondence(const std::filesystem::path &out_path,
                          const Skeleton &skeleton,
                          const Triangle_mesh &mesh) {
  std::ofstream out(out_path.string());
  if (!out) {
    std::cerr << "Error: cannot open " << out_path.string()
              << " for writing.\n";
    return false;
  }

  // The skeletonization algorithm computes a set of vertices on the input mesh
  // that correspond to each skeleton vertex. This function writes those
  // correspondences as line segments from the skeleton vertex to each
  // corresponding mesh.
  for (const Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
    for (const mesh_vertex_descriptor vd : skeleton[v].vertices) {
      out << "2 " << skeleton[v].point << " "
          << get(CGAL::vertex_point, mesh, vd) << "\n";
    }
  }
  return true;
}

} // namespace

bool skeletonize(Triangle_mesh &mesh, Skeleton &skeleton) {
  log_stage("3. Mean curvature flow skeletonization");

  // The mean curvature flow skeletonization algorithm a method that works
  // by simulating cloth on the surface of the mesh, which gradually contracts
  // it while preserving its topology.
  CGAL::extract_mean_curvature_flow_skeleton(mesh, skeleton);

  std::cout << "Skeleton vertices: " << boost::num_vertices(skeleton) << "\n";
  std::cout << "Skeleton edges: " << boost::num_edges(skeleton) << "\n";

  return boost::num_vertices(skeleton) > 0 && boost::num_edges(skeleton) > 0;
}

bool write_skeleton_outputs(const std::string &output_prefix, const Skeleton &skeleton,
                            const Triangle_mesh &mesh) {
  log_stage("3.1 Export skeleton artifacts");

  std::filesystem::path skeleton_polylines = output_prefix + "_skeleton.polylines.txt";
  std::filesystem::path skeleton_edges = output_prefix + "_skeleton_edges.txt";
  std::filesystem::path correspondence = output_prefix + "_correspondence.polylines.txt";

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
