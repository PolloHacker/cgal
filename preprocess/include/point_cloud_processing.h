#pragma once

#include "mesh_reconstruction.h"
#include "pipeline_config.h"

#include <string>
#include <vector>

/** \brief Loads an oriented point cloud from PLY and validates normals
 * availability. */
bool load_oriented_points(const std::string &input_path,
                          std::vector<mesh_reconstruction::Pwn> &points);

/** \brief Applies optional point-cloud filtering and computes average spacing.
 */
bool preprocess_points(std::vector<mesh_reconstruction::Pwn> &points,
                       const Pipeline_options &options,
                       double &average_spacing);
