// SPDX-License-Identifier: MIT
// Part of the C++ port of WNNC (Lin et al., ACM ToG 2024).
#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

#include "vec3.h"

namespace wnnc {

using Index = std::int64_t;

/// An octree over a fixed set of 3D points, stored as flat arrays ("structure
/// of arrays") so that traversal is cache-friendly and index-based.
///
/// Two invariants that the rest of the code relies on:
///  - Nodes are numbered in pre-order, so a parent always has a smaller index
///    than any node in its subtree. Bottom-up passes can therefore simply
///    iterate node indices in reverse.
///  - Each node owns a contiguous slice of `pointsInNode()`, containing the
///    indices of all points inside the node's cube (not just leaf points).
///
/// The root cube is centered at the origin with half-width 1; callers must
/// normalize their points into [-1, 1]^3 beforehand.
template <typename Scalar>
class Octree {
public:
    static constexpr int kNumChildren = 8;
    static constexpr Index kNoChild = -1;

    Octree(std::span<const Vec3<Scalar>> points, int maxDepth, Index maxPointsPerLeaf)
        : maxDepth_(maxDepth), maxPointsPerLeaf_(maxPointsPerLeaf) {
        assert(!points.empty());
        const auto pointCount = static_cast<Index>(points.size());
        pointIndices_.resize(static_cast<std::size_t>(pointCount));
        std::iota(pointIndices_.begin(), pointIndices_.end(), Index{0});

        // The recursion fills fixed 8-slot child arrays; they are compacted
        // into dense per-node child lists below, which keeps traversal reads
        // small and branch-free.
        std::vector<std::array<Index, kNumChildren>> childSlots;
        buildRecursive(points, childSlots, /*begin=*/0, /*end=*/pointCount,
                       /*center=*/Vec3<Scalar>{}, /*halfWidth=*/Scalar(1), /*depth=*/0);

        childBegin_.reserve(childSlots.size());
        childCount_.reserve(childSlots.size());
        childIndices_.reserve(childSlots.size() - 1);  // every non-root node is someone's child
        for (const std::array<Index, kNumChildren>& slots : childSlots) {
            childBegin_.push_back(static_cast<Index>(childIndices_.size()));
            for (const Index child : slots) {
                if (child != kNoChild) {
                    childIndices_.push_back(child);
                }
            }
            childCount_.push_back(static_cast<Index>(childIndices_.size()) - childBegin_.back());
        }
    }

    [[nodiscard]] Index nodeCount() const { return static_cast<Index>(halfWidths_.size()); }

    [[nodiscard]] Index leafCount() const {
        Index count = 0;
        for (const std::uint8_t flag : isLeaf_) {
            count += flag;
        }
        return count;
    }

    [[nodiscard]] int depth() const { return depth_; }

    [[nodiscard]] bool isLeaf(Index node) const {
        return isLeaf_[static_cast<std::size_t>(node)] != 0;
    }

    [[nodiscard]] Scalar halfWidth(Index node) const {
        return halfWidths_[static_cast<std::size_t>(node)];
    }

    /// The node's existing child indices (dense, between 0 and 8 entries).
    [[nodiscard]] std::span<const Index> children(Index node) const {
        const auto i = static_cast<std::size_t>(node);
        return std::span<const Index>(childIndices_)
            .subspan(static_cast<std::size_t>(childBegin_[i]),
                     static_cast<std::size_t>(childCount_[i]));
    }

    /// Indices of all points inside the node's cube.
    [[nodiscard]] std::span<const Index> pointsInNode(Index node) const {
        const auto i = static_cast<std::size_t>(node);
        return std::span<const Index>(pointIndices_)
            .subspan(static_cast<std::size_t>(rangeBegin_[i]),
                     static_cast<std::size_t>(rangeCount_[i]));
    }

    /// All point indices in tree (depth-first) order: spatially nearby points
    /// are nearby in this sequence, which makes it a cache-friendly order for
    /// issuing per-point tree traversals.
    [[nodiscard]] std::span<const Index> pointsInTreeOrder() const { return pointIndices_; }

    [[nodiscard]] Index pointCount(Index node) const {
        return rangeCount_[static_cast<std::size_t>(node)];
    }

private:
    Index buildRecursive(std::span<const Vec3<Scalar>> points,
                         std::vector<std::array<Index, kNumChildren>>& childSlots, Index begin,
                         Index end, const Vec3<Scalar>& center, Scalar halfWidth, int depth) {
        const Index node = static_cast<Index>(childSlots.size());
        std::array<Index, kNumChildren> noChildren;
        noChildren.fill(kNoChild);
        childSlots.push_back(noChildren);
        halfWidths_.push_back(halfWidth);
        isLeaf_.push_back(1);
        rangeBegin_.push_back(begin);
        rangeCount_.push_back(end - begin);
        depth_ = std::max(depth_, depth);

        if (depth >= maxDepth_ || end - begin <= maxPointsPerLeaf_) {
            return node;
        }
        isLeaf_[static_cast<std::size_t>(node)] = 0;

        // Stable counting sort of the node's point range into the 8 octants,
        // so each child ends up with a contiguous sub-range.
        std::array<Index, kNumChildren> octantCounts{};
        for (Index i = begin; i < end; ++i) {
            ++octantCounts[octantOf(points[static_cast<std::size_t>(pointIndices_[i])], center)];
        }

        std::array<Index, kNumChildren> octantBegins{};
        Index running = begin;
        for (int octant = 0; octant < kNumChildren; ++octant) {
            octantBegins[octant] = running;
            running += octantCounts[octant];
        }

        const std::vector<Index> unsortedRange(pointIndices_.begin() + begin,
                                               pointIndices_.begin() + end);
        std::array<Index, kNumChildren> writeCursors = octantBegins;
        for (const Index pointIndex : unsortedRange) {
            const int octant = octantOf(points[static_cast<std::size_t>(pointIndex)], center);
            pointIndices_[static_cast<std::size_t>(writeCursors[octant]++)] = pointIndex;
        }

        const Scalar childHalfWidth = halfWidth / Scalar(2);
        for (int octant = 0; octant < kNumChildren; ++octant) {
            if (octantCounts[octant] == 0) {
                continue;
            }
            const Vec3<Scalar> childCenter{
                center.x + ((octant & 1) ? childHalfWidth : -childHalfWidth),
                center.y + ((octant & 2) ? childHalfWidth : -childHalfWidth),
                center.z + ((octant & 4) ? childHalfWidth : -childHalfWidth)};
            const Index child =
                buildRecursive(points, childSlots, octantBegins[octant],
                               octantBegins[octant] + octantCounts[octant], childCenter,
                               childHalfWidth, depth + 1);
            childSlots[static_cast<std::size_t>(node)][octant] = child;
        }
        return node;
    }

    static int octantOf(const Vec3<Scalar>& point, const Vec3<Scalar>& center) {
        return (point.x >= center.x ? 1 : 0) | (point.y >= center.y ? 2 : 0) |
               (point.z >= center.z ? 4 : 0);
    }

    int maxDepth_;
    Index maxPointsPerLeaf_;
    int depth_ = 0;

    std::vector<Index> childIndices_;  // dense child lists, sliced per node below
    std::vector<Index> childBegin_;
    std::vector<Index> childCount_;
    std::vector<Scalar> halfWidths_;
    std::vector<std::uint8_t> isLeaf_;  // std::uint8_t: vector<bool> is not addressable
    std::vector<Index> rangeBegin_;
    std::vector<Index> rangeCount_;
    std::vector<Index> pointIndices_;  // permutation of [0, N); sliced by the ranges above
};

}  // namespace wnnc
