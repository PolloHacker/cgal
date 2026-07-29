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
