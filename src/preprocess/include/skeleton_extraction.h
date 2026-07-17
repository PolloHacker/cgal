#pragma once

#include "mesh_reconstruction.h"

#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <string>
#include <memory>

/** \brief Skeleton type used by the MCF skeletonization routine. */
using Skeletonization =
    CGAL::Mean_curvature_flow_skeletonization<
        mesh_reconstruction::Triangle_mesh>;
using Skeleton = Skeletonization::Skeleton;

/** \brief Parameters for configuration of Mean Curvature Flow skeletonization. */
struct SkeletonizationParameters {
    double max_triangle_angle = 110.0;
    double min_edge_length = 0.0; // 0.0 or negative triggers CGAL's default auto-calculation
    std::size_t max_iterations = 500;
    double area_variation_factor = 0.0001;
    double quality_speed_tradeoff = 0.1;
    bool is_medially_centered = true;
    double medially_centered_speed_tradeoff = 0.2;
};

/** \brief Interface for skeletonization operations. */
class ISkeletonizer {
public:
    virtual ~ISkeletonizer() = default;
    virtual bool skeletonize(mesh_reconstruction::Triangle_mesh &mesh, Skeleton &skeleton) = 0;
};

/** \brief Concrete Mean Curvature Flow skeletonizer implementing ISkeletonizer. */
class MeanCurvatureFlowSkeletonizer : public ISkeletonizer {
public:
    explicit MeanCurvatureFlowSkeletonizer(const SkeletonizationParameters &params = SkeletonizationParameters())
        : params_(params) {}

    bool skeletonize(mesh_reconstruction::Triangle_mesh &mesh, Skeleton &skeleton) override;

    /** \brief Helper for testing: creates and configures the CGAL MCF object. */
    std::unique_ptr<Skeletonization> create_mcf(const mesh_reconstruction::Triangle_mesh &mesh) const;

private:
    SkeletonizationParameters params_;
};

/** \brief Runs mean-curvature-flow skeleton extraction on a closed triangle
 * mesh using default parameters. */
bool skeletonize(mesh_reconstruction::Triangle_mesh &mesh, Skeleton &skeleton);

/** \brief Writes all skeleton artifacts to disk. */
bool write_skeleton_outputs(const std::string &output_prefix, const Skeleton &skeleton,
                            const mesh_reconstruction::Triangle_mesh &mesh);
