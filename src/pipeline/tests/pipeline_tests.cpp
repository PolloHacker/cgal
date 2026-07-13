#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../poisson_recon_wrapper.h"
#include <cmath>
#include <vector>
#include <iostream>

// Helper to generate a sphere point cloud
std::vector<PipelinePoint> generate_sphere(double radius, int u_segments, int v_segments) {
    std::vector<PipelinePoint> points;
    for (int i = 0; i <= u_segments; ++i) {
        double u = i * M_PI / u_segments; // 0 to pi
        for (int j = 0; j < v_segments; ++j) {
            double v = j * 2 * M_PI / v_segments; // 0 to 2pi
            double x = radius * std::sin(u) * std::cos(v);
            double y = radius * std::sin(u) * std::sin(v);
            double z = radius * std::cos(u);
            // Normal points outwards from center
            double nx = x / radius;
            double ny = y / radius;
            double nz = z / radius;
            points.push_back({x, y, z, nx, ny, nz});
        }
    }
    return points;
}

TEST_CASE("Poisson Reconstruction and Trimming Unit Tests") {
    // Generate a sphere with 400 points
    auto sphere_points = generate_sphere(5.0, 20, 20);

    PoissonParams poisson_params;
    poisson_params.depth = 6; // low depth for fast unit test
    poisson_params.verbose = false;

    SUBCASE("Stage 3: Screened Poisson Reconstruction (Untrimmed)") {
        PipelineMesh mesh = run_poisson_reconstruction(sphere_points, poisson_params);
        
        // Assert that a valid mesh is reconstructed
        CHECK(mesh.vertices.size() > 0);
        CHECK(mesh.faces.size() > 0);

        // Assert all vertices have non-zero coordinates and valid density values
        for (const auto& vertex : mesh.vertices) {
            CHECK(std::isfinite(vertex.x));
            CHECK(std::isfinite(vertex.y));
            CHECK(std::isfinite(vertex.z));
            CHECK(vertex.value >= 0.0);
        }
    }

    SUBCASE("Stage 4: Surface Trimming (Optional)") {
        PipelineMesh untrimmed_mesh = run_poisson_reconstruction(sphere_points, poisson_params);
        REQUIRE(untrimmed_mesh.vertices.size() > 0);

        TrimParams trim_params;
        trim_params.trimThreshold = 2.0; // trim value
        trim_params.islandAreaRatio = 0.001;
        trim_params.removeIslands = true;
        trim_params.verbose = false;

        PipelineMesh trimmed_mesh = trim_mesh(untrimmed_mesh, trim_params);

        // Assert that trimmed mesh has fewer or equal vertices
        CHECK(trimmed_mesh.vertices.size() <= untrimmed_mesh.vertices.size());
        
        // Assert all trimmed vertices are valid
        for (const auto& vertex : trimmed_mesh.vertices) {
            CHECK(std::isfinite(vertex.x));
            CHECK(std::isfinite(vertex.y));
            CHECK(std::isfinite(vertex.z));
            CHECK(vertex.value >= 0.0);
        }
    }

    SUBCASE("Solver Combinations: Degree and Boundary Types") {
        // Test all degree (1, 2) and boundary type (1=Free, 2=Dirichlet, 3=Neumann) configurations
        for (int degree : {1, 2}) {
            for (int bType : {1, 2, 3}) {
                PoissonParams params;
                params.depth = 5; // even smaller depth for fast testing
                params.degree = degree;
                params.bType = bType;
                params.verbose = false;

                PipelineMesh mesh;
                CHECK_NOTHROW(mesh = run_poisson_reconstruction(sphere_points, params));
                
                // Verify the mesh contains reconstructed vertices and faces
                CHECK(mesh.vertices.size() > 0);
                CHECK(mesh.faces.size() > 0);

                // Verify the values are valid
                for (const auto& v : mesh.vertices) {
                    CHECK(std::isfinite(v.x));
                    CHECK(std::isfinite(v.value));
                }
            }
        }
    }
}
