#include "doctest.h"
#include "skeleton_extraction.h"
#include "mesh_reconstruction.h"

using namespace mesh_reconstruction;

namespace {
Triangle_mesh create_closed_cube() {
    Triangle_mesh mesh;
    
    // Add 8 vertices of a cube
    auto v0 = mesh.add_vertex(Point(0.0, 0.0, 0.0));
    auto v1 = mesh.add_vertex(Point(1.0, 0.0, 0.0));
    auto v2 = mesh.add_vertex(Point(1.0, 1.0, 0.0));
    auto v3 = mesh.add_vertex(Point(0.0, 1.0, 0.0));
    auto v4 = mesh.add_vertex(Point(0.0, 0.0, 1.0));
    auto v5 = mesh.add_vertex(Point(1.0, 0.0, 1.0));
    auto v6 = mesh.add_vertex(Point(1.0, 1.0, 1.0));
    auto v7 = mesh.add_vertex(Point(0.0, 1.0, 1.0));
    
    // Bottom face
    mesh.add_face(v3, v2, v1);
    mesh.add_face(v3, v1, v0);
    // Front face
    mesh.add_face(v0, v1, v5);
    mesh.add_face(v0, v5, v4);
    // Right face
    mesh.add_face(v1, v2, v6);
    mesh.add_face(v1, v6, v5);
    // Back face
    mesh.add_face(v2, v3, v7);
    mesh.add_face(v2, v7, v6);
    // Left face
    mesh.add_face(v3, v0, v4);
    mesh.add_face(v3, v4, v7);
    // Top face
    mesh.add_face(v4, v5, v6);
    mesh.add_face(v4, v6, v7);
    
    return mesh;
}
} // namespace

TEST_CASE("SkeletonizationParameters structure initialization") {
    SkeletonizationParameters params;
    CHECK(params.max_triangle_angle == doctest::Approx(110.0));
    CHECK(params.min_edge_length == doctest::Approx(0.0));
    CHECK(params.max_iterations == 500);
    CHECK(params.area_variation_factor == doctest::Approx(0.0001));
    CHECK(params.quality_speed_tradeoff == doctest::Approx(0.1));
    CHECK(params.is_medially_centered == true);
    CHECK(params.medially_centered_speed_tradeoff == doctest::Approx(0.2));
}

TEST_CASE("MeanCurvatureFlowSkeletonizer default parameters skeletonize success") {
    Triangle_mesh mesh = create_closed_cube();
    bool normalized = normalize_mesh_for_skeletonization(mesh, true);
    REQUIRE(normalized);
    
    MeanCurvatureFlowSkeletonizer skeletonizer;
    Skeleton skeleton;
    bool success = skeletonizer.skeletonize(mesh, skeleton);
    
    CHECK(success);
    CHECK(boost::num_vertices(skeleton) > 0);
    CHECK(boost::num_edges(skeleton) > 0);
}

TEST_CASE("MeanCurvatureFlowSkeletonizer configures CGAL Skeletonization correctly") {
    Triangle_mesh mesh = create_closed_cube();
    
    SkeletonizationParameters params;
    params.max_triangle_angle = 120.0;
    params.min_edge_length = 0.05;
    params.max_iterations = 300;
    params.area_variation_factor = 0.0005;
    params.quality_speed_tradeoff = 0.15;
    params.is_medially_centered = false;
    params.medially_centered_speed_tradeoff = 0.25;
    
    MeanCurvatureFlowSkeletonizer skeletonizer(params);
    auto mcf = skeletonizer.create_mcf(mesh);
    
    REQUIRE(mcf != nullptr);
    CHECK(mcf->max_triangle_angle() == doctest::Approx(120.0));
    CHECK(mcf->min_edge_length() == doctest::Approx(0.05));
    CHECK(mcf->max_iterations() == 300);
    CHECK(mcf->area_variation_factor() == doctest::Approx(0.0005));
    CHECK(mcf->quality_speed_tradeoff() == doctest::Approx(0.15));
    CHECK(mcf->is_medially_centered() == false);
    CHECK(mcf->medially_centered_speed_tradeoff() == doctest::Approx(0.25));
}
