#include "mesh_reconstruction.h"
#include "validation.h"

#if __has_include(<CGAL/IO/polygon_mesh_io.h>)
#include <CGAL/IO/polygon_mesh_io.h>
#elif __has_include(<CGAL/boost/graph/IO/polygon_mesh_io.h>)
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>
#else
#error "polygon_mesh_io header not found in this CGAL installation"
#endif
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/poisson_surface_reconstruction.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;

namespace mesh_reconstruction {

bool normalize_mesh_for_skeletonization(Triangle_mesh &mesh,
                                        const bool keep_largest_component) {
  using mesh_halfedge_descriptor =
      boost::graph_traits<Triangle_mesh>::halfedge_descriptor;

  if (!CGAL::is_valid_polygon_mesh(mesh)) {
    std::cerr << "Error: reconstructed mesh is not a valid polygon mesh.\n";
    return false;
  }

  if (!CGAL::is_triangle_mesh(mesh)) {
    std::cout << "Triangulating faces...\n";
    if (!PMP::triangulate_faces(mesh)) {
      std::cerr << "Error: triangulate_faces() failed.\n";
      return false;
    }
  }

  if (!CGAL::is_triangle_mesh(mesh)) {
    std::cerr
        << "Error: mesh is still not triangulated after triangulation pass.\n";
    return false;
  }

  if (keep_largest_component) {
    const std::size_t removed_cc =
        PMP::keep_largest_connected_components(mesh, 1);
    if (removed_cc > 0) {
      std::cout << "Removed connected components: " << removed_cc << "\n";
    }
  }

  if (!CGAL::is_closed(mesh)) {
    std::vector<mesh_halfedge_descriptor> boundary_cycles;
    PMP::extract_boundary_cycles(mesh, std::back_inserter(boundary_cycles));
    std::size_t filled_holes = 0;

    for (const mesh_halfedge_descriptor h : boundary_cycles) {
      if (!CGAL::is_border(h, mesh)) {
        continue;
      }

      const std::size_t face_count_before = num_faces(mesh);
      PMP::triangulate_hole(
          mesh, h,
          CGAL::parameters::use_delaunay_triangulation(true)
              .use_2d_constrained_delaunay_triangulation(true));

      if (num_faces(mesh) > face_count_before) {
        ++filled_holes;
      }
    }

    if (filled_holes > 0) {
      std::cout << "Filled boundary cycles: " << filled_holes << "\n";
    }
  }

  if (!CGAL::is_closed(mesh)) {
    const std::size_t stitched = PMP::stitch_borders(mesh);
    if (stitched > 0) {
      std::cout << "Stitched border halfedge pairs: " << stitched << "\n";
    }
  }

  if (!CGAL::is_closed(mesh)) {
    std::cerr << "Error: mesh is not closed. Skeletonization requires a closed "
                 "surface.\n";
    return false;
  }

  std::cout << "Mesh check passed: triangulated + closed.\n";
  return true;
}

bool write_mesh_ply(const std::filesystem::path &output_mesh_path,
                    const Triangle_mesh &mesh) {
  if (!CGAL::IO::write_polygon_mesh(
          output_mesh_path.string(), mesh,
          CGAL::parameters::stream_precision(17).use_binary_mode(false))) {
    std::cerr << "Error: cannot write reconstructed mesh to "
              << output_mesh_path.string() << "\n";
    return false;
  }

  return true;
}

} // namespace mesh_reconstruction
