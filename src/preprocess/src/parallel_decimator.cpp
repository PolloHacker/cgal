#include "parallel_decimator.h"

#include <iostream>
#include <vector>

namespace mesh_reconstruction {

// Declarations of factory helpers
std::unique_ptr<IDecimator> create_tbb_cpu_decimator();
std::unique_ptr<IDecimator> create_cuda_decimator();

std::unique_ptr<IDecimator> create_decimator(const DecimationOptions& options) {
    if (options.enable_cuda) {
        auto cuda_dec = create_cuda_decimator();
        if (cuda_dec) {
            std::cout << "[DecimatorFactory] Selected CUDA GPU Decimator Backend.\n";
            return cuda_dec;
        }
        std::cout << "[DecimatorFactory] CUDA unavailable or disabled. Falling back to Multi-Threaded TBB CPU Decimator.\n";
    }
    
    std::cout << "[DecimatorFactory] Selected Multi-Threaded TBB CPU Decimator Backend.\n";
    return create_tbb_cpu_decimator();
}

bool decimate_point_set(Point_set& point_set, double min_distance, bool enable_cuda) {
    if (point_set.empty() || min_distance <= 0.0) {
        return false;
    }

    // Extract Point_set to std::vector<Point3D>
    std::vector<Point3D> input_points;
    input_points.reserve(point_set.size());

    for (auto p_it = point_set.begin(); p_it != point_set.end(); ++p_it) {
        Point3D pt;
        const auto& p = point_set.point(*p_it);
        pt.x = CGAL::to_double(p.x());
        pt.y = CGAL::to_double(p.y());
        pt.z = CGAL::to_double(p.z());

        if (point_set.has_normal_map()) {
            const auto& n = point_set.normal(*p_it);
            pt.nx = CGAL::to_double(n.x());
            pt.ny = CGAL::to_double(n.y());
            pt.nz = CGAL::to_double(n.z());
        }

        input_points.push_back(pt);
    }

    DecimationOptions options;
    options.min_distance = min_distance;
    options.enable_cuda = enable_cuda;

    auto decimator = create_decimator(options);
    DecimationResult result = decimator->decimate(input_points, options);

    if (result.points.empty()) {
        std::cerr << "Error: Parallel decimation produced 0 points.\n";
        return false;
    }

    // Rebuild CGAL Point_set
    Point_set decimated_set;
    if (point_set.has_normal_map()) {
        decimated_set.add_normal_map();
    }
    for (const auto& pt : result.points) {
        auto idx = decimated_set.insert(Point(pt.x, pt.y, pt.z));
        if (point_set.has_normal_map()) {
            decimated_set.normal(idx) = Vector(pt.nx, pt.ny, pt.nz);
        }
    }

    std::cout << "Parallel decimation complete (" 
              << (result.used_cuda ? "CUDA GPU" : "Multi-Thread CPU TBB")
              << "): " << point_set.size() << " -> " << decimated_set.size() << " points.\n";

    point_set = std::move(decimated_set);
    return true;
}

} // namespace mesh_reconstruction
