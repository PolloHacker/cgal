#include "parallel_decimator.h"

#include <tbb/concurrent_unordered_set.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>

#include <chrono>
#include <iostream>

namespace mesh_reconstruction {

class TbbCpuDecimator : public IDecimator {
public:
    DecimationResult decimate(std::span<const Point3D> points, const DecimationOptions& options) override {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        DecimationResult result;
        result.original_count = points.size();
        result.used_cuda = false;

        if (points.empty()) {
            return result;
        }

        tbb::concurrent_unordered_set<VoxelKey, VoxelKeyHash> occupied_voxels;
        tbb::concurrent_vector<Point3D> decimated_vector;
        decimated_vector.reserve(points.size() / 4);

        const double min_dist = options.min_distance;

        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, points.size(), 10000),
            [&](const tbb::blocked_range<std::size_t>& r) {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& pt = points[i];
                    VoxelKey key = VoxelKey::from_point(pt, min_dist);
                    
                    auto res = occupied_voxels.insert(key);
                    if (res.second) {
                        // First point to claim this voxel
                        decimated_vector.push_back(pt);
                    }
                }
            }
        );

        result.points.assign(decimated_vector.begin(), decimated_vector.end());
        result.decimated_count = result.points.size();

        auto end_time = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::cout << "[TbbCpuDecimator] Decimated " << result.original_count 
                  << " -> " << result.decimated_count << " points in " 
                  << result.execution_time_ms << " ms\n";

        return result;
    }
};

std::unique_ptr<IDecimator> create_tbb_cpu_decimator() {
    return std::make_unique<TbbCpuDecimator>();
}

} // namespace mesh_reconstruction
