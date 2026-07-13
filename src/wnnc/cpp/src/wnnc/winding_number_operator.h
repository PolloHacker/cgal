// SPDX-License-Identifier: MIT
// Part of the C++ port of WNNC (Lin et al., ACM ToG 2024).
#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "octree.h"
#include "parallel.h"
#include "treecode_kernels.h"
#include "vec3.h"

namespace wnnc {

/// Per-node summaries used by the treecode's far-field approximation: the sum
/// of the point attributes in the node's subtree, a weighted centroid (the
/// "representative point") and the total attribute weight.
template <typename Scalar, typename Attr>
struct NodeAggregates {
    std::vector<Attr> attrSums;
    std::vector<Vec3<Scalar>> repPoints;
    std::vector<Scalar> weights;
};

namespace detail {

/// Weight of a point attribute when averaging representative points.
template <typename Scalar>
[[nodiscard]] Scalar attrWeight(const Vec3<Scalar>& attr) {
    return attr.norm();
}

template <typename Scalar>
[[nodiscard]] Scalar attrWeight(Scalar attr) {
    return std::abs(attr);
}

}  // namespace detail

/// Fast evaluation of the winding-number operators A, A^T and G over a fixed
/// point set, using a Barnes-Hut style treecode: distant subtrees are
/// approximated by a single dipole/source at their representative point.
///
/// This is the C++ counterpart of the Python `WindingNumberTreecode` class
/// (and of the reference CPU/CUDA kernels in `wn_treecode`).
template <typename Scalar>
class WindingNumberOperator {
public:
    /// The reference implementation's ALLOWED_MAX_DEPTH; bounds the size of
    /// the fixed traversal stack below.
    static constexpr int kMaxTreeDepth = 15;
    /// Fastest measured setting; also slightly *more* accurate than the
    /// reference's 1, because leaves are always evaluated exactly.
    static constexpr Index kDefaultMaxPointsPerLeaf = 4;

    /// Splitting stops at `maxPointsPerLeaf`. The reference implementation
    /// uses 1; larger leaves are evaluated exactly (never approximated), so
    /// accuracy only improves, and small batches traverse faster than
    /// singleton leaves.
    explicit WindingNumberOperator(std::vector<Vec3<Scalar>> points,
                                   int maxTreeDepth = kMaxTreeDepth,
                                   Index maxPointsPerLeaf = kDefaultMaxPointsPerLeaf)
        : points_(std::move(points)),
          tree_(points_, std::min(maxTreeDepth, kMaxTreeDepth), maxPointsPerLeaf) {
        if (points_.empty()) {
            throw std::invalid_argument("WindingNumberOperator: empty point set");
        }
        // Precompute each node's squared far-field distance: a subtree is
        // "far" when the query is at least kFarFieldFactor node half-widths
        // from its representative point (the reference implementation's
        // TREECODE_THRESHOLD * 2).
        constexpr Scalar kFarFieldFactor = Scalar(4);
        farFieldDistanceSq_.reserve(static_cast<std::size_t>(tree_.nodeCount()));
        for (Index node = 0; node < tree_.nodeCount(); ++node) {
            const Scalar farDistance = kFarFieldFactor * tree_.halfWidth(node);
            farFieldDistanceSq_.push_back(farDistance * farDistance);
        }
    }

    [[nodiscard]] Index pointCount() const { return static_cast<Index>(points_.size()); }
    [[nodiscard]] const std::vector<Vec3<Scalar>>& points() const { return points_; }
    [[nodiscard]] const Octree<Scalar>& tree() const { return tree_; }

    /// A mu: the winding number induced by the dipoles `mu` (one per point),
    /// evaluated at every point. `smoothingWidths` holds one positive width
    /// per point.
    [[nodiscard]] std::vector<Scalar> applyA(std::span<const Vec3<Scalar>> mu,
                                             std::span<const Scalar> smoothingWidths) const {
        return applyOperator<Scalar>(mu, smoothingWidths, [](const auto&... args) {
            return kernels::windingNumberTerm(args...);
        });
    }

    /// A^T values: the transpose of `applyA`, mapping one scalar per point
    /// back to a vector per point.
    [[nodiscard]] std::vector<Vec3<Scalar>> applyATransposed(
        std::span<const Scalar> values, std::span<const Scalar> smoothingWidths) const {
        return applyOperator<Vec3<Scalar>>(values, smoothingWidths, [](const auto&... args) {
            return kernels::windingNumberTransposeTerm(args...);
        });
    }

    /// G mu: the (negated) gradient of the winding-number field induced by
    /// the dipoles `mu`, evaluated at every point.
    [[nodiscard]] std::vector<Vec3<Scalar>> applyG(std::span<const Vec3<Scalar>> mu,
                                                   std::span<const Scalar> smoothingWidths) const {
        return applyOperator<Vec3<Scalar>>(mu, smoothingWidths, [](const auto&... args) {
            return kernels::windingNumberGradientTerm(args...);
        });
    }

    /// Bottom-up aggregation of per-point attributes into per-node summaries.
    /// `Attr` is Vec3<Scalar> for A/G (dipoles) and Scalar for A^T (sources).
    template <typename Attr>
    [[nodiscard]] NodeAggregates<Scalar, Attr> aggregateToNodes(
        std::span<const Attr> pointAttrs) const {
        const Index nodeCount = tree_.nodeCount();
        NodeAggregates<Scalar, Attr> aggregates;
        aggregates.attrSums.assign(static_cast<std::size_t>(nodeCount), Attr{});
        aggregates.repPoints.assign(static_cast<std::size_t>(nodeCount), Vec3<Scalar>{});
        aggregates.weights.assign(static_cast<std::size_t>(nodeCount), Scalar(0));

        // Leaves aggregate directly over the points they contain.
        parallelFor(nodeCount, [&](Index node) {
            if (!tree_.isLeaf(node)) {
                return;
            }
            Attr attrSum{};
            Vec3<Scalar> weightedPositionSum{};
            Vec3<Scalar> positionSum{};
            Scalar totalWeight{};
            for (const Index point : tree_.pointsInNode(node)) {
                const auto p = static_cast<std::size_t>(point);
                const Scalar weight = detail::attrWeight(pointAttrs[p]);
                attrSum += pointAttrs[p];
                weightedPositionSum += points_[p] * weight;
                positionSum += points_[p];
                totalWeight += weight;
            }
            const auto n = static_cast<std::size_t>(node);
            aggregates.attrSums[n] = attrSum;
            aggregates.weights[n] = totalWeight;
            // With zero total weight the node contributes nothing anyway
            // (all attributes in it are zero), so any representative point
            // works; use the plain centroid.
            aggregates.repPoints[n] =
                totalWeight > Scalar(0)
                    ? weightedPositionSum * (Scalar(1) / totalWeight)
                    : positionSum * (Scalar(1) / Scalar(tree_.pointCount(node)));
        });

        // Internal nodes aggregate their children. Pre-order numbering means
        // children always have larger indices than their parent, so a single
        // reverse sweep visits every child before its parent.
        for (Index node = nodeCount - 1; node >= 0; --node) {
            if (tree_.isLeaf(node)) {
                continue;
            }
            Attr attrSum{};
            Vec3<Scalar> weightedRepSum{};
            Vec3<Scalar> repSum{};
            Scalar totalWeight{};
            for (const Index child : tree_.children(node)) {
                const auto c = static_cast<std::size_t>(child);
                attrSum += aggregates.attrSums[c];
                weightedRepSum += aggregates.repPoints[c] * aggregates.weights[c];
                repSum += aggregates.repPoints[c];
                totalWeight += aggregates.weights[c];
            }
            const auto childCount = static_cast<Scalar>(tree_.children(node).size());
            const auto n = static_cast<std::size_t>(node);
            aggregates.attrSums[n] = attrSum;
            aggregates.weights[n] = totalWeight;
            aggregates.repPoints[n] = totalWeight > Scalar(0)
                                          ? weightedRepSum * (Scalar(1) / totalWeight)
                                          : repSum * (Scalar(1) / childCount);
        }
        return aggregates;
    }

private:
    /// Shared implementation of the three operators: aggregates the point
    /// attributes into the nodes, then evaluates `term(diff, attr, width)`
    /// over each query's far nodes and near points. Queries are issued in
    /// tree order, so consecutive queries traverse nearly the same nodes and
    /// the hot node data stays in cache.
    template <typename Result, typename Attr, typename TermFn>
    [[nodiscard]] std::vector<Result> applyOperator(std::span<const Attr> pointAttrs,
                                                    std::span<const Scalar> smoothingWidths,
                                                    const TermFn& term) const {
        assert(pointAttrs.size() == points_.size() &&
               smoothingWidths.size() == points_.size());
        const NodeAggregates<Scalar, Attr> aggregates = aggregateToNodes(pointAttrs);
        const std::span<const Index> treeOrder = tree_.pointsInTreeOrder();

        std::vector<Result> result(points_.size());
        parallelFor(pointCount(), [&](Index slot) {
            const auto i = static_cast<std::size_t>(treeOrder[static_cast<std::size_t>(slot)]);
            const Vec3<Scalar> query = points_[i];
            const Scalar width = smoothingWidths[i];
            Result sum{};
            forEachContribution(
                query, aggregates.repPoints,
                [&](Index node) {
                    const auto n = static_cast<std::size_t>(node);
                    sum += term(query - aggregates.repPoints[n], aggregates.attrSums[n], width);
                },
                [&](Index point) {
                    const auto p = static_cast<std::size_t>(point);
                    sum += term(query - points_[p], pointAttrs[p], width);
                });
            result[i] = sum;
        });
        return result;
    }

    /// Depth-first treecode traversal for one query point. Calls
    /// `onFarNode(node)` for subtrees far enough to be approximated by their
    /// aggregate, and `onNearPoint(point)` for each point of a nearby leaf.
    template <typename FarNodeFn, typename NearPointFn>
    void forEachContribution(const Vec3<Scalar>& query,
                             std::span<const Vec3<Scalar>> nodeRepPoints,
                             const FarNodeFn& onFarNode,
                             const NearPointFn& onNearPoint) const {
        // Upper bound on the stack: descending one path expands one node per
        // level, leaving at most 7 pending siblings per level, plus the node
        // being expanded.
        constexpr std::size_t kStackCapacity =
            static_cast<std::size_t>(kMaxTreeDepth + 1) * (Octree<Scalar>::kNumChildren - 1) + 1;
        std::array<Index, kStackCapacity> stack;
        std::size_t stackSize = 0;
        stack[stackSize++] = 0;  // root

        while (stackSize > 0) {
            const Index node = stack[--stackSize];
            const Vec3<Scalar> toRepPoint =
                query - nodeRepPoints[static_cast<std::size_t>(node)];

            if (toRepPoint.squaredNorm() > farFieldDistanceSq_[static_cast<std::size_t>(node)]) {
                onFarNode(node);
            } else if (!tree_.isLeaf(node)) {
                for (const Index child : tree_.children(node)) {
                    assert(stackSize < stack.size());
                    stack[stackSize++] = child;
                }
            } else {
                for (const Index point : tree_.pointsInNode(node)) {
                    onNearPoint(point);
                }
            }
        }
    }

    std::vector<Vec3<Scalar>> points_;  // must precede tree_: the octree views it
    Octree<Scalar> tree_;
    std::vector<Scalar> farFieldDistanceSq_;
};

}  // namespace wnnc
