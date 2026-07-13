#pragma once

#include "mesh_reconstruction.h"
#include <string>

namespace mesh_reconstruction {

/**
 * \brief Loads a PLY file into a CGAL Point_set_3, with optional spatial (voxel-grid) subsampling.
 * \details Automatically handles ASCII and binary (Little Endian) PLY formats. If spatial
 *          subsampling is enabled, it uses an on-the-fly voxel grid filtering approach to
 *          minimize memory usage by loading only one point per voxel.
 * \param filepath Path to the input PLY file.
 * \param points Destination point set to populate.
 * \param enable_spatial_subsampling If true, applies voxel-grid subsampling on-the-fly.
 * \param min_distance Voxel edge length for spatial subsampling.
 * \return true if loading and optional subsampling succeed, false otherwise.
 */
bool load_ply(const std::string &filepath,
              Point_set &points,
              bool enable_spatial_subsampling = false,
              double min_distance = 0.1);

/**
 * \brief Writes a CGAL Point_set_3 to a PLY file.
 * \details Serializes all registered properties (normals, colors, intensity) to the PLY file.
 * \param filepath Path to the output PLY file.
 * \param points Point set to serialize.
 * \param binary If true, writes in binary format; otherwise, writes in ASCII.
 * \return true if writing succeeds, false otherwise.
 */
bool write_ply(const std::string &filepath,
               const Point_set &points,
               bool binary = true);

} // namespace mesh_reconstruction
