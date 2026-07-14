#pragma once

#include <filesystem>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/property_map.h>
#include <CGAL/Point_set_3.h>

namespace mesh_reconstruction {
/** \brief Kernel used by the reconstruction and skeletonization pipeline. */
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;

/** \brief Concurrency tag for parallel execution. */
using Concurrency_tag = CGAL::Parallel_if_available_tag;

/** \brief 3D point type used in the input point cloud. */
using Point = Kernel::Point_3;

/** \brief 3D vector type used for oriented normals. */
using Vector = Kernel::Vector_3;

/** \brief Pair containing one point and its associated normal. */
using Pwn = std::pair<Point, Vector>;

/** \brief Property map extracting the point from a point-normal pair. */
using Point_map = CGAL::First_of_pair_property_map<Pwn>;

/** \brief Property map extracting the normal from a point-normal pair. */
using Normal_map = CGAL::Second_of_pair_property_map<Pwn>;

/** \brief Unified point set type preserving all attributes (normals, colors, intensity). */
using Point_set = CGAL::Point_set_3<Point>;

/** \brief Triangle mesh type produced by Poisson reconstruction. */
using Triangle_mesh = CGAL::Surface_mesh<Point>;

/**
 * \brief Enforces skeletonization preconditions on a reconstructed mesh.
 * \details This routine validates the mesh, triangulates faces, optionally
 * keeps only the largest connected component, fills boundary holes, and
 * stitches borders so the result is closed. 
 * \param mesh Reconstructed mesh to normalize. 
 * \param keep_largest_component If `true`, keeps only the largest
 * connected component. 
 * \return `true` when the mesh is triangulated and closed.
 */
bool normalize_mesh_for_skeletonization(Triangle_mesh &mesh,
                                        bool keep_largest_component);

/**
 * \brief Writes a mesh snapshot to an ASCII PLY file for visualization.
 * \param output_mesh_path Destination path.
 * \param mesh Mesh to serialize.
 * \return `true` when serialization succeeds.
 */
bool write_mesh_ply(const std::filesystem::path &output_mesh_path,
                    const Triangle_mesh &mesh,
                    bool binary = true);

} // namespace mesh_reconstruction
