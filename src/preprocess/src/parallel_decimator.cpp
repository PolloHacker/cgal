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

} // namespace mesh_reconstruction
