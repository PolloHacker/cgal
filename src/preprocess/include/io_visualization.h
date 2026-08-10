#pragma once

#include "mesh_reconstruction.h"

#include <filesystem>
#include <string>
#include <vector>

/** \brief Emits a stage label to make the pipeline trace easy to follow. */
void log_stage(const std::string &label);

/** \brief Exports a point-cloud snapshot for visualization and logs the stage.
 */
bool write_point_stage_visualization(
    const std::filesystem::path &out_path,
    const mesh_reconstruction::Point_set &points,
    const std::string &stage_label);

/** \brief Exports a mesh snapshot for visualization and logs the stage. */
bool write_mesh_stage_visualization(const std::filesystem::path &out_path,
                                    const mesh_reconstruction::Triangle_mesh &mesh,
                                    const std::string &stage_label);
