#pragma once

#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>
#include <span>
#include <string>

#include "mesh_reconstruction.h"

namespace mesh_reconstruction {

struct Point3D {
    double x = 0.0, y = 0.0, z = 0.0;
    double nx = 0.0, ny = 0.0, nz = 0.0;
    uint8_t r = 255, g = 255, b = 255;
    float intensity = 0.0f;
};

struct VoxelKey {
    int32_t ix = 0;
    int32_t iy = 0;
    int32_t iz = 0;

    [[nodiscard]] constexpr bool operator==(const VoxelKey& other) const noexcept {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }

    [[nodiscard]] static VoxelKey from_point(const Point3D& p, double min_distance) noexcept {
        const double inv = 1.0 / min_distance;
        return {
            static_cast<int32_t>(std::floor(p.x * inv)),
            static_cast<int32_t>(std::floor(p.y * inv)),
            static_cast<int32_t>(std::floor(p.z * inv))
        };
    }

    [[nodiscard]] constexpr uint64_t pack() const noexcept {
        const uint64_t ux = static_cast<uint64_t>(ix + 0x100000) & 0x1FFFFFULL;
        const uint64_t uy = static_cast<uint64_t>(iy + 0x100000) & 0x1FFFFFULL;
        const uint64_t uz = static_cast<uint64_t>(iz + 0x100000) & 0x1FFFFFULL;
        return (ux << 42) | (uy << 21) | uz;
    }

    [[nodiscard]] uint64_t hash() const noexcept {
        return (static_cast<uint64_t>(ix) * 73856093ULL) ^
               (static_cast<uint64_t>(iy) * 19349663ULL) ^
               (static_cast<uint64_t>(iz) * 83492791ULL);
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const noexcept {
        return static_cast<std::size_t>(k.hash());
    }
};

struct DecimationOptions {
    double min_distance = 0.05;
    std::size_t chunk_size = 10000000; // 10M points per chunk
    bool enable_cuda = true;
    std::size_t max_vram_bytes = 2048ULL * 1024ULL * 1024ULL; // 2GB
};

struct DecimationResult {
    std::vector<Point3D> points;
    std::size_t original_count = 0;
    std::size_t decimated_count = 0;
    double execution_time_ms = 0.0;
    bool used_cuda = false;
};

/**
 * \brief Pure virtual seam interface for Parallel Decimation backends (Brooks-lint Deep Module pattern).
 */
class IDecimator {
public:
    virtual ~IDecimator() = default;
    virtual DecimationResult decimate(std::span<const Point3D> points, const DecimationOptions& options) = 0;
};

/**
 * \brief Factory function returning optimal decimator backend (CUDA GPU if available, or TBB CPU).
 */
std::unique_ptr<IDecimator> create_decimator(const DecimationOptions& options);

/**
 * \brief High-level helper function to decimate a CGAL Point_set in place.
 */
bool decimate_point_set(Point_set& point_set, double min_distance, bool enable_cuda = true);

} // namespace mesh_reconstruction
