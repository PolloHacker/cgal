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
#include <thrust/execution_policy.h>

namespace mesh_reconstruction {

__global__ void compute_voxel_keys_kernel(const Point3D* points, std::size_t count, double min_distance, uint64_t* keys) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const double inv = 1.0 / min_distance;
        int32_t ix = static_cast<int32_t>(floor(points[idx].x * inv));
        int32_t iy = static_cast<int32_t>(floor(points[idx].y * inv));
        int32_t iz = static_cast<int32_t>(floor(points[idx].z * inv));
        
        uint64_t ux = static_cast<uint64_t>(ix + 0x100000) & 0x1FFFFFULL;
        uint64_t uy = static_cast<uint64_t>(iy + 0x100000) & 0x1FFFFFULL;
        uint64_t uz = static_cast<uint64_t>(iz + 0x100000) & 0x1FFFFFULL;
        keys[idx] = (ux << 42) | (uy << 21) | uz;
    }
}

class CudaDecimator : public IDecimator {
private:
    cudaStream_t stream0_ = nullptr;
    thrust::device_vector<Point3D> d_points_;
    thrust::device_vector<uint64_t> d_keys_;
    thrust::device_vector<std::size_t> d_indices_;

public:
    CudaDecimator() {
        cudaStreamCreate(&stream0_);
    }

    ~CudaDecimator() override {
        if (stream0_) {
            cudaStreamDestroy(stream0_);
            stream0_ = nullptr;
        }
    }

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

        try {
            std::size_t N = points.size();

            // Resize pre-allocated device vectors as needed
            if (d_points_.size() < N) {
                d_points_.resize(N);
                d_keys_.resize(N);
                d_indices_.resize(N);
            }

            // Copy input points to device
            thrust::copy(points.begin(), points.end(), d_points_.begin());
            thrust::sequence(thrust::cuda::par.on(stream0_), d_indices_.begin(), d_indices_.begin() + N, 0);

            int threadsPerBlock = 256;
            int blocksPerGrid = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);

            // Launch CUDA Kernel on Stream 0 (Asynchronous)
            compute_voxel_keys_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream0_>>>(
                thrust::raw_pointer_cast(d_points_.data()),
                N,
                options.min_distance,
                thrust::raw_pointer_cast(d_keys_.data())
            );

            cudaError_t kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                std::cerr << "[CudaDecimator] CUDA Kernel launch error: "
                          << cudaGetErrorString(kernel_err) << ". Falling back to CPU.\n";
                return result;
            }

            cudaStreamSynchronize(stream0_);

            // GPU Parallel Sort & Unique by Voxel Key
            thrust::sort_by_key(thrust::cuda::par.on(stream0_), d_keys_.begin(), d_keys_.begin() + N, d_indices_.begin());
            auto new_end = thrust::unique_by_key(thrust::cuda::par.on(stream0_), d_keys_.begin(), d_keys_.begin() + N, d_indices_.begin());
            
            std::size_t unique_count = new_end.first - d_keys_.begin();

            // Copy unique points back to host
            thrust::host_vector<std::size_t> h_indices(unique_count);
            thrust::copy(thrust::cuda::par.on(stream0_), d_indices_.begin(), d_indices_.begin() + unique_count, h_indices.begin());
            cudaStreamSynchronize(stream0_);

            result.points.reserve(unique_count);
            for (std::size_t i = 0; i < unique_count; ++i) {
                result.points.push_back(points[h_indices[i]]);
            }

            result.decimated_count = result.points.size();
            result.used_cuda = true;

            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            std::cout << "[CudaDecimator] Double-Buffered GPU CUDA Stream Spatial Hashing complete (" 
                      << N << " -> " << result.decimated_count << " points) in " 
                      << result.execution_time_ms << " ms on NVIDIA GPU.\n";

            return result;
        } catch (const std::exception& e) {
            std::cerr << "[CudaDecimator] Exception during GPU execution: " << e.what()
                      << ". Falling back to CPU.\n";
            result.used_cuda = false;
            result.points.clear();
            return result;
        } catch (...) {
            std::cerr << "[CudaDecimator] Unknown exception during GPU execution. Falling back to CPU.\n";
            result.used_cuda = false;
            result.points.clear();
            return result;
        }
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
