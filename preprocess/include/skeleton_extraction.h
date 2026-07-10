#pragma once

#include "mesh_reconstruction.h"

#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <string>

/** \brief Skeleton type used by the MCF skeletonization routine. */
using Skeletonization =
    CGAL::Mean_curvature_flow_skeletonization<
        mesh_reconstruction::Triangle_mesh>;
using Skeleton = Skeletonization::Skeleton;

/** \brief Runs mean-curvature-flow skeleton extraction on a closed triangle
 * mesh. */
bool skeletonize(mesh_reconstruction::Triangle_mesh &mesh, Skeleton &skeleton);

/** \brief Writes all skeleton artifacts to disk. */
bool write_skeleton_outputs(const std::string &output_prefix, const Skeleton &skeleton,
                            const mesh_reconstruction::Triangle_mesh &mesh);
