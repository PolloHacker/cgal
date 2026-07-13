#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../poisson_recon_wrapper.h"
#include "wnnc/normal_solver.h"
#include "wnnc/winding_number_operator.h"
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

// Helper to generate simple double-precision points for WNNC
std::vector<wnnc::Vec3<double>> generate_sphere_coords(double radius, int u_segments, int v_segments) {
    std::vector<wnnc::Vec3<double>> points;
    for (int i = 0; i <= u_segments; ++i) {
        double u = i * M_PI / u_segments;
        for (int j = 0; j < v_segments; ++j) {
            double v = j * 2 * M_PI / v_segments;
            double x = radius * std::sin(u) * std::cos(v);
            double y = radius * std::sin(u) * std::sin(v);
            double z = radius * std::cos(u);
            points.push_back({x, y, z});
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

TEST_CASE("CUDA-Accelerated WNNC Normal Estimation Unit Tests") {
    // Generate a sphere with 420 points
    auto coords = generate_sphere_coords(1.0, 20, 20);

    // Instantiate CUDA-accelerated winding number operator
    // Using max depth = 10, max points per leaf = 1
    wnnc::WindingNumberOperator<double> windingNumber(coords, 10, 1);

    CHECK(windingNumber.pointCount() == (wnnc::Index)coords.size());

    // Setup WNNC solver settings
    wnnc::SolverSettings settings;
    settings.iterations = 10; // 10 iterations for unit test speed
    settings.widthMin = 0.05;
    settings.widthMax = 0.2;

    SUBCASE("CUDA WNNC Normal Estimation Execution") {
        std::vector<wnnc::Vec3<double>> estimated_normals;
        
        // Assert that the estimation finishes without throwing CUDA errors
        CHECK_NOTHROW(estimated_normals = wnnc::estimateNormals(windingNumber, settings));

        REQUIRE(estimated_normals.size() == coords.size());

        // Verify that estimated normals are unit vectors and reasonably aligned
        for (size_t i = 0; i < estimated_normals.size(); ++i) {
            double norm = estimated_normals[i].norm();
            CHECK(norm == doctest::Approx(1.0).epsilon(1e-4));

            // Check that the normal points roughly in the same direction as the sphere position
            // (since the sphere is centered at origin, pos . normal should be positive)
            double dot = coords[i].x * estimated_normals[i].x +
                         coords[i].y * estimated_normals[i].y +
                         coords[i].z * estimated_normals[i].z;
            
            // Allow some deviation due to approximation, but it must be positive (outwards pointing)
            CHECK(dot > 0.0);
        }
    }
}
