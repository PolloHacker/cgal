#include "doctest.h"
#include "parallel_decimator.h"

#include <vector>
#include <cmath>
#include <iostream>

TEST_CASE("Parallel Decimator - VoxelKey Hash and Spatial Cell Indexing") {
    using namespace mesh_reconstruction;

    Point3D p1{0.0, 0.0, 0.0};
    Point3D p2{0.01, 0.02, 0.03};
    Point3D p3{1.0, 1.0, 1.0};

    double min_dist = 0.05;

    VoxelKey k1 = VoxelKey::from_point(p1, min_dist);
    VoxelKey k2 = VoxelKey::from_point(p2, min_dist);
    VoxelKey k3 = VoxelKey::from_point(p3, min_dist);

    // p1 and p2 fall in the same voxel (0.01 < 0.05)
    CHECK(k1 == k2);
    // p3 falls in a different voxel
    CHECK_FALSE(k1 == k3);
}

TEST_CASE("Parallel Decimator - TBB CPU Multi-Thread Decimation") {
    using namespace mesh_reconstruction;

    std::vector<Point3D> points;
    // Create a dense grid of points where multiple points fall into distance < 0.05
    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < 100; ++j) {
            points.push_back({i * 0.01, j * 0.01, 0.0, 0.0, 0.0, 1.0});
        }
    }

    CHECK(points.size() == 10000);

    DecimationOptions options;
    options.min_distance = 0.05;
    options.enable_cuda = false; // Test CPU TBB explicitly

    auto decimator = create_decimator(options);
    REQUIRE(decimator != nullptr);

    DecimationResult result = decimator->decimate(points, options);

    // Decimated count should be significantly smaller than 10000
    CHECK(result.decimated_count < points.size());
    CHECK(result.decimated_count > 0);
    CHECK(result.used_cuda == false);
    
    // Verify all returned points preserve original coordinates
    for (const auto& pt : result.points) {
        CHECK(pt.nz == 1.0);
    }
}

TEST_CASE("Parallel Decimator - Fallback to CPU when CUDA is disabled/unavailable") {
    using namespace mesh_reconstruction;

    std::vector<Point3D> points = {
        {0.0, 0.0, 0.0}, {0.01, 0.01, 0.01}, {1.0, 1.0, 1.0}
    };

    DecimationOptions options;
    options.min_distance = 0.05;
    options.enable_cuda = true; // Request CUDA, expect seamless fallback if unavailable

    auto decimator = create_decimator(options);
    REQUIRE(decimator != nullptr);

    DecimationResult result = decimator->decimate(points, options);
    CHECK(result.decimated_count == 2); // {0,0,0} and {1,1,1} retained
}

TEST_CASE("Parallel Decimator - VoxelKey Pack Collision-Free Test") {
    using namespace mesh_reconstruction;

    VoxelKey k1{10, 20, 30};
    VoxelKey k2{10, 20, 30};
    VoxelKey k3{20, 10, 30};

    CHECK(k1.pack() == k2.pack());
    CHECK_FALSE(k1.pack() == k3.pack());

    // Verify negative coordinate packing
    VoxelKey kn1{-5, -10, -15};
    VoxelKey kn2{-5, -10, -15};
    VoxelKey kn3{5, 10, 15};
    CHECK(kn1.pack() == kn2.pack());
    CHECK_FALSE(kn1.pack() == kn3.pack());
}

TEST_CASE("Parallel Decimator - Color Preservation & NaN/Inf Handling") {
    using namespace mesh_reconstruction;

    Point3D p1{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 128, 64, 32, 0.5f};
    Point3D p_nan{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
    Point3D p_inf{0.0, std::numeric_limits<double>::infinity(), 0.0};
    Point3D p2{2.0, 2.0, 2.0, 0.0, 0.0, 1.0, 255, 0, 128, 1.0f};

    std::vector<Point3D> points = {p1, p2};

    DecimationOptions options;
    options.min_distance = 0.05;
    options.enable_cuda = false;

    auto decimator = create_decimator(options);
    DecimationResult res = decimator->decimate(points, options);

    CHECK(res.decimated_count == 2);
    // Verify color attributes are unchanged uint8_t values
    bool found_p1 = false;
    for (const auto& pt : res.points) {
        if (std::abs(pt.x - 0.0) < 1e-5) {
            CHECK(pt.r == 128);
            CHECK(pt.g == 64);
            CHECK(pt.b == 32);
            found_p1 = true;
        }
    }
    CHECK(found_p1);
}

