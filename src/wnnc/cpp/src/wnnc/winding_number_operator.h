// SPDX-License-Identifier: MIT
// Pimpl header for CUDA-accelerated WindingNumberOperator
#pragma once

#include <vector>
#include <span>
#include <memory>
#include "vec3.h"

namespace wnnc {

using Index = std::int64_t;

template <typename Scalar>
class WindingNumberOperator {
public:
    explicit WindingNumberOperator(std::vector<Vec3<Scalar>> points, int maxTreeDepth = 15, Index maxPointsPerLeaf = 4);
    ~WindingNumberOperator();

    // Disable copy to prevent double-freeing GPU memory
    WindingNumberOperator(const WindingNumberOperator&) = delete;
    WindingNumberOperator& operator=(const WindingNumberOperator&) = delete;
    
    WindingNumberOperator(WindingNumberOperator&&) noexcept;
    WindingNumberOperator& operator=(WindingNumberOperator&&) noexcept;

    [[nodiscard]] Index pointCount() const;
    [[nodiscard]] const std::vector<Vec3<Scalar>>& points() const;
    [[nodiscard]] Index nodeCount() const;
    [[nodiscard]] Index leafCount() const;
    [[nodiscard]] int depth() const;

    [[nodiscard]] std::vector<Scalar> applyA(std::span<const Vec3<Scalar>> mu, std::span<const Scalar> smoothingWidths) const;
    [[nodiscard]] std::vector<Vec3<Scalar>> applyATransposed(std::span<const Scalar> values, std::span<const Scalar> smoothingWidths) const;
    [[nodiscard]] std::vector<Vec3<Scalar>> applyG(std::span<const Vec3<Scalar>> mu, std::span<const Scalar> smoothingWidths) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace wnnc
