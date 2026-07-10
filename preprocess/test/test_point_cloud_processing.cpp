#include "doctest.h"
#include "point_cloud_processing.h"
#include "pipeline_config.h"

using namespace mesh_reconstruction;

TEST_CASE("Testing point cloud processing pipeline") {
    Point_set points;
    points.add_normal_map();
    
    // Add 6 points
    Point_set::Index idx;
    idx = *(points.insert(Point(0.0, 0.0, 0.0))); points.normal(idx) = Vector(0.0, 0.0, 1.0);
    idx = *(points.insert(Point(1.0, 0.0, 0.0))); points.normal(idx) = Vector(0.0, 0.0, 1.0);
    idx = *(points.insert(Point(0.0, 1.0, 0.0))); points.normal(idx) = Vector(0.0, 0.0, 1.0);
    idx = *(points.insert(Point(1.0, 1.0, 0.0))); points.normal(idx) = Vector(0.0, 0.0, 1.0);
    idx = *(points.insert(Point(0.5, 0.5, 0.0))); points.normal(idx) = Vector(0.0, 0.0, 1.0);
    idx = *(points.insert(Point(10.0, 10.0, 10.0))); points.normal(idx) = Vector(0.0, 0.0, 1.0); // outlier
    
    SUBCASE("Outlier removal and smoothing") {
        Pipeline_options options;
        options.outlier_neighbors = 3;
        options.outlier_percent = 15.0; // remove the outlier
        options.enable_wlop = false;
        options.enable_smoothing = true;
        options.smoothing_neighbors = 3;
        
        double avg_spacing = 0.0;
        bool preprocess_ok = preprocess_points(points, options, avg_spacing);
        CHECK(preprocess_ok);
        CHECK(points.size() < 6); // at least the outlier point should be removed
        CHECK(avg_spacing > 0.0);
    }
    
    SUBCASE("WLOP downsampling") {
        Pipeline_options options;
        options.outlier_neighbors = 3;
        options.outlier_percent = 0.0;
        options.enable_wlop = true;
        options.wlop_retain_percent = 50.0; // keep 50% of 6 points = 3 points
        options.wlop_neighbor_radius = -1.0;
        options.wlop_iterations = 10;
        options.wlop_require_uniform_sampling = false;
        options.enable_smoothing = false;
        
        double avg_spacing = 0.0;
        bool preprocess_ok = preprocess_points(points, options, avg_spacing);
        CHECK(preprocess_ok);
        CHECK(points.size() == 3);
    }
}
