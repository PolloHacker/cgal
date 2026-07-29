#include "parallel_decimator.h"

#include <vector>
#include <iostream>

namespace mesh_reconstruction {

class ChunkedStreamer {
public:
    static DecimationResult stream_and_decimate(IDecimator& decimator,
                                                 std::span<const Point3D> points,
                                                 const DecimationOptions& options) {
        if (points.size() <= options.chunk_size) {
            return decimator.decimate(points, options);
        }

        std::cout << "[ChunkedStreamer] Point cloud has " << points.size() 
                  << " points, streaming in chunks of " << options.chunk_size << " points...\n";

        DecimationResult combined_result;
        combined_result.original_count = points.size();
        
        std::size_t offset = 0;
        std::size_t chunk_idx = 0;

        while (offset < points.size()) {
            std::size_t count = std::min(options.chunk_size, points.size() - offset);
            std::span<const Point3D> chunk = points.subspan(offset, count);

            DecimationResult chunk_res = decimator.decimate(chunk, options);
            combined_result.points.insert(combined_result.points.end(),
                                          chunk_res.points.begin(),
                                          chunk_res.points.end());

            combined_result.execution_time_ms += chunk_res.execution_time_ms;
            combined_result.used_cuda = chunk_res.used_cuda;
            offset += count;
            chunk_idx++;
        }

        // Secondary pass to deduplicate boundaries between chunks
        DecimationResult final_pass = decimator.decimate(combined_result.points, options);
        final_pass.original_count = points.size();
        final_pass.execution_time_ms += combined_result.execution_time_ms;
        
        return final_pass;
    }
};

} // namespace mesh_reconstruction
