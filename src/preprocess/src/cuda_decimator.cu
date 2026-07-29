#include "parallel_decimator.h"

#include <iostream>
#include <vector>
#include <chrono>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>

namespace mesh_reconstruction {

__global__ void compute_voxel_keys_kernel(const Point3D* points, std::size_t count, double min_distance, uint64_t* keys) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const double inv = 1.0 / min_distance;
        int64_t ix = static_cast<int64_t>(floor(points[idx].x * inv));
        int64_t iy = static_cast<int64_t>(floor(points[idx].y * inv));
        int64_t iz = static_cast<int64_t>(floor(points[idx].z * inv));
        
        uint64_t hash = (static_cast<uint64_t>(ix) * 73856093ULL) ^
                       (static_cast<uint64_t>(iy) * 19349663ULL) ^
                       (static_cast<uint64_t>(iz) * 83492791ULL);
        keys[idx] = hash;
    }
}

class CudaDecimator : public IDecimator {
public:
    DecimationResult decimate(std::span<const Point3D> points, const DecimationOptions& options) override {
        auto start_time = std::chrono::high_resolution_clock::now();

        DecimationResult result;
        result.original_count = points.size();
        result.used_cuda = false;

        if (points.empty()) {
            return result;
        }

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            std::cerr << "[CudaDecimator] CUDA hardware unavailable. Falling back to CPU.\n";
            return result;
        }

        std::size_t N = points.size();

        // Device allocations
        thrust::device_vector<Point3D> d_points(points.begin(), points.end());
        thrust::device_vector<uint64_t> d_keys(N);
        thrust::device_vector<std::size_t> d_indices(N);
        thrust::sequence(d_indices.begin(), d_indices.end(), 0);

        int threadsPerBlock = 256;
        int blocksPerGrid = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);

        // Launch CUDA Kernel
        compute_voxel_keys_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_points.data()),
            N,
            options.min_distance,
            thrust::raw_pointer_cast(d_keys.data())
        );
        cudaDeviceSynchronize();

        // GPU Parallel Sort & Unique by Voxel Key
        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());
        auto new_end = thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());
        
        std::size_t unique_count = new_end.first - d_keys.begin();

        // Copy unique points back to host
        thrust::host_vector<std::size_t> h_indices(d_indices.begin(), d_indices.begin() + unique_count);
        
        result.points.reserve(unique_count);
        for (std::size_t i = 0; i < unique_count; ++i) {
            result.points.push_back(points[h_indices[i]]);
        }

        result.decimated_count = result.points.size();
        result.used_cuda = true;

        auto end_time = std::chrono::high_resolution_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::cout << "[CudaDecimator] GPU CUDA Spatial Hashing complete (" << N << " -> " 
                  << result.decimated_count << " points) in " 
                  << result.execution_time_ms << " ms on NVIDIA GPU.\n";

        return result;
    }
};

std::unique_ptr<IDecimator> create_cuda_decimator() {
    return std::make_unique<CudaDecimator>();
}

} // namespace mesh_reconstruction

#else

namespace mesh_reconstruction {
std::unique_ptr<IDecimator> create_cuda_decimator() {
    return nullptr;
}
} // namespace mesh_reconstruction

#endif
