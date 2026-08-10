// SPDX-License-Identifier: MIT
#include "wnnc/winding_number_operator.h"
#include "wn_treecode_cpu.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace wnnc {

template <typename Scalar>
struct WindingNumberOperator<Scalar>::Impl {
    Index point_count = 0;
    std::vector<Vec3<Scalar>> points_host;

    // Tree dimensions
    signedindex_t num_nodes = 0;
    signedindex_t num_leaves = 0;
    signedindex_t tree_depth = 0;

    // CPU buffers for the tree (all single precision float)
    std::vector<signedindex_t> node_parent_list;
    std::vector<signedindex_t> node_children_list;
    std::unique_ptr<bool[]> node_is_leaf_list;
    std::vector<signedindex_t> num_points_in_node;
    std::vector<signedindex_t> node2point_indexstart;
    std::vector<float> node_half_w_list;
    std::vector<signedindex_t> node2point_index;
    std::vector<float> flat_points;

    // Temporary/helper CPU buffers to eliminate allocator overhead
    mutable std::vector<float> temp_mu;                  // point_count * 3
    mutable std::vector<float> temp_weights;             // point_count
    mutable std::vector<float> temp_widths;              // point_count
    std::unique_ptr<bool[]> temp_scattered_mask;       // num_nodes
    std::unique_ptr<bool[]> temp_next_to_scatter_mask;  // num_nodes
    mutable std::vector<float> temp_node_attrs;          // num_nodes * 3
    mutable std::vector<float> temp_node_reppoints;      // num_nodes * 3
    mutable std::vector<float> temp_node_weights;        // num_nodes
    mutable std::vector<float> temp_out_attrs;           // point_count * 3

    // CPU vectors to eliminate vector reallocation overhead
    mutable std::vector<float> host_flat_mu;
    mutable std::vector<float> host_weights;
    mutable std::vector<float> host_widths;

    Impl(std::vector<Vec3<Scalar>> points, int maxTreeDepth, Index maxPointsPerLeaf)
        : point_count(static_cast<Index>(points.size())), points_host(std::move(points)) {
        if (point_count == 0) {
            throw std::invalid_argument("WindingNumberOperator: empty point set");
        }

        // 1. Flatten and cast coordinates to float
        flat_points.resize(point_count * 3);
        for (size_t i = 0; i < (size_t)point_count; ++i) {
            flat_points[3 * i + 0] = static_cast<float>(points_host[i].x);
            flat_points[3 * i + 1] = static_cast<float>(points_host[i].y);
            flat_points[3 * i + 2] = static_cast<float>(points_host[i].z);
        }

        // 2. Build the octree using float coords on CPU
        std::vector<signedindex_t> point_indices(point_count);
        for (signedindex_t i = 0; i < point_count; ++i) {
            point_indices[i] = i;
        }

        signedindex_t cur_node_index = 0;
        OctNode<float>* root = build_tree_cpu_recursive<float>(
            flat_points.data(),
            point_indices,
            nullptr,
            0.0f, 0.0f, 0.0f, // center
            1.0f,             // half width (normalized to unit cube)
            0,                // depth
            cur_node_index,
            maxTreeDepth,
            static_cast<signedindex_t>(maxPointsPerLeaf)
        );

        compute_tree_attributes<float>(root, num_nodes, num_leaves, tree_depth);

        // 3. Serialize tree to CPU host vectors
        node_parent_list.resize(num_nodes);
        node_children_list.resize(num_nodes * NUM_OCT_CHILDREN);
        node_is_leaf_list = std::make_unique<bool[]>(num_nodes);
        node_half_w_list.resize(num_nodes);
        num_points_in_node.resize(num_nodes);
        node2point_indexstart.resize(num_nodes);

        serialize_tree_recursive<float>(
            root,
            node_parent_list.data(),
            node_children_list.data(),
            node_is_leaf_list.get(),
            node_half_w_list.data(),
            num_points_in_node.data(),
            node2point_indexstart.data(),
            node2point_index
        );

        free_tree_recursive<float>(root);

        // Allocate temporary/helper buffers
        temp_mu.resize(point_count * 3);
        temp_weights.resize(point_count);
        temp_widths.resize(point_count);
        temp_scattered_mask = std::make_unique<bool[]>(num_nodes);
        temp_next_to_scatter_mask = std::make_unique<bool[]>(num_nodes);
        temp_node_attrs.resize(num_nodes * 3);
        temp_node_reppoints.resize(num_nodes * 3);
        temp_node_weights.resize(num_nodes);
        temp_out_attrs.resize(point_count * 3);

        // Resize mutable host buffers
        host_flat_mu.resize(point_count * 3);
        host_weights.resize(point_count);
        host_widths.resize(point_count);
    }

    std::vector<Scalar> applyA(std::span<const Vec3<Scalar>> mu, std::span<const Scalar> smoothingWidths) const {
        for (size_t i = 0; i < (size_t)point_count; ++i) {
            float mx = static_cast<float>(mu[i].x);
            float my = static_cast<float>(mu[i].y);
            float mz = static_cast<float>(mu[i].z);
            host_flat_mu[3 * i + 0] = mx;
            host_flat_mu[3 * i + 1] = my;
            host_flat_mu[3 * i + 2] = mz;
            host_weights[i] = std::sqrt(mx * mx + my * my + mz * mz);
            host_widths[i] = static_cast<float>(smoothingWidths[i]);
        }

        std::memcpy(temp_mu.data(), host_flat_mu.data(), point_count * 3 * sizeof(float));
        std::memcpy(temp_weights.data(), host_weights.data(), point_count * sizeof(float));
        std::memcpy(temp_widths.data(), host_widths.data(), point_count * sizeof(float));

        std::memset(temp_scattered_mask.get(), 0, num_nodes * sizeof(bool));
        std::memset(temp_next_to_scatter_mask.get(), 0, num_nodes * sizeof(bool));
        std::memset(temp_node_attrs.data(), 0, num_nodes * 3 * sizeof(float));
        std::memset(temp_node_reppoints.data(), 0, num_nodes * 3 * sizeof(float));
        std::memset(temp_node_weights.data(), 0, num_nodes * sizeof(float));

        scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher<float>(
            node_parent_list.data(),
            flat_points.data(),
            temp_weights.data(),
            temp_mu.data(),
            node2point_index.data(),
            node2point_indexstart.data(),
            num_points_in_node.data(),
            node_is_leaf_list.get(),
            temp_scattered_mask.get(),
            temp_node_attrs.data(),
            temp_node_reppoints.data(),
            temp_node_weights.data(),
            3, // attr_dim = 3
            num_nodes
        );

        for (signedindex_t depth = tree_depth - 1; depth >= 0; depth--) {
            find_next_to_scatter_cpu_kernel_launcher<float>(
                node_children_list.data(),
                node_is_leaf_list.get(),
                temp_scattered_mask.get(),
                temp_next_to_scatter_mask.get(),
                node2point_index.data(),
                num_nodes
            );

            scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher<float>(
                node_parent_list.data(),
                node_children_list.data(),
                flat_points.data(),
                temp_weights.data(),
                temp_mu.data(),
                node2point_index.data(),
                node2point_indexstart.data(),
                num_points_in_node.data(),
                node_is_leaf_list.get(),
                temp_scattered_mask.get(),
                temp_next_to_scatter_mask.get(),
                temp_node_attrs.data(),
                temp_node_reppoints.data(),
                temp_node_weights.data(),
                3, // attr_dim = 3
                num_nodes
            );
        }

        std::memset(temp_out_attrs.data(), 0, point_count * sizeof(float));

        multiply_by_A_cpu_kernel_launcher<float>(
            flat_points.data(),
            temp_widths.data(),
            flat_points.data(),
            temp_mu.data(),
            node2point_index.data(),
            node2point_indexstart.data(),
            node_children_list.data(),
            temp_node_attrs.data(),
            node_is_leaf_list.get(),
            node_half_w_list.data(),
            temp_node_reppoints.data(),
            num_points_in_node.data(),
            temp_out_attrs.data(),
            point_count,
            true
        );

        std::vector<Scalar> result(point_count);
        for (std::int64_t i = 0; i < (std::int64_t)point_count; ++i) {
            result[i] = static_cast<Scalar>(temp_out_attrs[i]);
        }
        return result;
    }

    std::vector<Vec3<Scalar>> applyATransposed(std::span<const Scalar> values, std::span<const Scalar> smoothingWidths) const {
        for (std::int64_t i = 0; i < (std::int64_t)point_count; ++i) {
            float val = static_cast<float>(values[i]);
            host_flat_mu[i] = val;
            host_weights[i] = std::abs(val);
            host_widths[i] = static_cast<float>(smoothingWidths[i]);
        }

        std::memcpy(temp_mu.data(), host_flat_mu.data(), point_count * sizeof(float));
        std::memcpy(temp_weights.data(), host_weights.data(), point_count * sizeof(float));
        std::memcpy(temp_widths.data(), host_widths.data(), point_count * sizeof(float));

        std::memset(temp_scattered_mask.get(), 0, num_nodes * sizeof(bool));
        std::memset(temp_next_to_scatter_mask.get(), 0, num_nodes * sizeof(bool));
        std::memset(temp_node_attrs.data(), 0, num_nodes * sizeof(float));
        std::memset(temp_node_reppoints.data(), 0, num_nodes * 3 * sizeof(float));
        std::memset(temp_node_weights.data(), 0, num_nodes * sizeof(float));

        scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher<float>(
            node_parent_list.data(),
            flat_points.data(),
            temp_weights.data(),
            temp_mu.data(),
            node2point_index.data(),
            node2point_indexstart.data(),
            num_points_in_node.data(),
            node_is_leaf_list.get(),
            temp_scattered_mask.get(),
            temp_node_attrs.data(),
            temp_node_reppoints.data(),
            temp_node_weights.data(),
            1, // attr_dim = 1
            num_nodes
        );

        for (signedindex_t depth = tree_depth - 1; depth >= 0; depth--) {
            find_next_to_scatter_cpu_kernel_launcher<float>(
                node_children_list.data(),
                node_is_leaf_list.get(),
                temp_scattered_mask.get(),
                temp_next_to_scatter_mask.get(),
                node2point_index.data(),
                num_nodes
            );

            scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher<float>(
                node_parent_list.data(),
                node_children_list.data(),
                flat_points.data(),
                temp_weights.data(),
                temp_mu.data(),
                node2point_index.data(),
                node2point_indexstart.data(),
                num_points_in_node.data(),
                node_is_leaf_list.get(),
                temp_scattered_mask.get(),
                temp_next_to_scatter_mask.get(),
                temp_node_attrs.data(),
                temp_node_reppoints.data(),
                temp_node_weights.data(),
                1, // attr_dim = 1
                num_nodes
            );
        }

        std::memset(temp_out_attrs.data(), 0, point_count * 3 * sizeof(float));

        multiply_by_AT_cpu_kernel_launcher<float>(
            flat_points.data(),
            temp_widths.data(),
            flat_points.data(),
            temp_mu.data(),
            node2point_index.data(),
            node2point_indexstart.data(),
            node_children_list.data(),
            temp_node_attrs.data(),
            node_is_leaf_list.get(),
            node_half_w_list.data(),
            temp_node_reppoints.data(),
            num_points_in_node.data(),
            temp_out_attrs.data(),
            point_count
        );

        std::vector<Vec3<Scalar>> result(point_count);
        for (std::int64_t i = 0; i < (std::int64_t)point_count; ++i) {
            result[i] = {
                static_cast<Scalar>(temp_out_attrs[3 * i + 0]),
                static_cast<Scalar>(temp_out_attrs[3 * i + 1]),
                static_cast<Scalar>(temp_out_attrs[3 * i + 2])
            };
        }
        return result;
    }

    std::vector<Vec3<Scalar>> applyG(std::span<const Vec3<Scalar>> mu, std::span<const Scalar> smoothingWidths) const {
        for (std::int64_t i = 0; i < (std::int64_t)point_count; ++i) {
            float mx = static_cast<float>(mu[i].x);
            float my = static_cast<float>(mu[i].y);
            float mz = static_cast<float>(mu[i].z);
            host_flat_mu[3 * i + 0] = mx;
            host_flat_mu[3 * i + 1] = my;
            host_flat_mu[3 * i + 2] = mz;
            host_weights[i] = std::sqrt(mx * mx + my * my + mz * mz);
            host_widths[i] = static_cast<float>(smoothingWidths[i]);
        }

        std::memcpy(temp_mu.data(), host_flat_mu.data(), point_count * 3 * sizeof(float));
        std::memcpy(temp_weights.data(), host_weights.data(), point_count * sizeof(float));
        std::memcpy(temp_widths.data(), host_widths.data(), point_count * sizeof(float));

        std::memset(temp_scattered_mask.get(), 0, num_nodes * sizeof(bool));
        std::memset(temp_next_to_scatter_mask.get(), 0, num_nodes * sizeof(bool));
        std::memset(temp_node_attrs.data(), 0, num_nodes * 3 * sizeof(float));
        std::memset(temp_node_reppoints.data(), 0, num_nodes * 3 * sizeof(float));
        std::memset(temp_node_weights.data(), 0, num_nodes * sizeof(float));

        scatter_point_attrs_to_nodes_leaf_cpu_kernel_launcher<float>(
            node_parent_list.data(),
            flat_points.data(),
            temp_weights.data(),
            temp_mu.data(),
            node2point_index.data(),
            node2point_indexstart.data(),
            num_points_in_node.data(),
            node_is_leaf_list.get(),
            temp_scattered_mask.get(),
            temp_node_attrs.data(),
            temp_node_reppoints.data(),
            temp_node_weights.data(),
            3, // attr_dim = 3
            num_nodes
        );

        for (signedindex_t depth = tree_depth - 1; depth >= 0; depth--) {
            find_next_to_scatter_cpu_kernel_launcher<float>(
                node_children_list.data(),
                node_is_leaf_list.get(),
                temp_scattered_mask.get(),
                temp_next_to_scatter_mask.get(),
                node2point_index.data(),
                num_nodes
            );

            scatter_point_attrs_to_nodes_nonleaf_cpu_kernel_launcher<float>(
                node_parent_list.data(),
                node_children_list.data(),
                flat_points.data(),
                temp_weights.data(),
                temp_mu.data(),
                node2point_index.data(),
                node2point_indexstart.data(),
                num_points_in_node.data(),
                node_is_leaf_list.get(),
                temp_scattered_mask.get(),
                temp_next_to_scatter_mask.get(),
                temp_node_attrs.data(),
                temp_node_reppoints.data(),
                temp_node_weights.data(),
                3, // attr_dim = 3
                num_nodes
            );
        }

        std::memset(temp_out_attrs.data(), 0, point_count * 3 * sizeof(float));

        multiply_by_G_cpu_kernel_launcher<float>(
            flat_points.data(),
            temp_widths.data(),
            flat_points.data(),
            temp_mu.data(),
            node2point_index.data(),
            node2point_indexstart.data(),
            node_children_list.data(),
            temp_node_attrs.data(),
            node_is_leaf_list.get(),
            node_half_w_list.data(),
            temp_node_reppoints.data(),
            num_points_in_node.data(),
            temp_out_attrs.data(),
            point_count
        );

        std::vector<Vec3<Scalar>> result(point_count);
        for (std::int64_t i = 0; i < (std::int64_t)point_count; ++i) {
            result[i] = {
                static_cast<Scalar>(temp_out_attrs[3 * i + 0]),
                static_cast<Scalar>(temp_out_attrs[3 * i + 1]),
                static_cast<Scalar>(temp_out_attrs[3 * i + 2])
            };
        }
        return result;
    }
};

// -------------------------------------------------------------------------
// WindingNumberOperator Template Outer Definitions
// -------------------------------------------------------------------------

template <typename Scalar>
WindingNumberOperator<Scalar>::WindingNumberOperator(std::vector<Vec3<Scalar>> points, int maxTreeDepth, Index maxPointsPerLeaf)
    : impl_(std::make_unique<Impl>(std::move(points), maxTreeDepth, maxPointsPerLeaf)) {}

template <typename Scalar>
WindingNumberOperator<Scalar>::~WindingNumberOperator() = default;

template <typename Scalar>
WindingNumberOperator<Scalar>::WindingNumberOperator(WindingNumberOperator&&) noexcept = default;

template <typename Scalar>
WindingNumberOperator<Scalar>& WindingNumberOperator<Scalar>::operator=(WindingNumberOperator&&) noexcept = default;

template <typename Scalar>
Index WindingNumberOperator<Scalar>::pointCount() const {
    return impl_->point_count;
}

template <typename Scalar>
const std::vector<Vec3<Scalar>>& WindingNumberOperator<Scalar>::points() const {
    return impl_->points_host;
}

template <typename Scalar>
Index WindingNumberOperator<Scalar>::nodeCount() const {
    return impl_->num_nodes;
}

template <typename Scalar>
Index WindingNumberOperator<Scalar>::leafCount() const {
    return impl_->num_leaves;
}

template <typename Scalar>
int WindingNumberOperator<Scalar>::depth() const {
    return static_cast<int>(impl_->tree_depth);
}

template <typename Scalar>
std::vector<Scalar> WindingNumberOperator<Scalar>::applyA(std::span<const Vec3<Scalar>> mu, std::span<const Scalar> smoothingWidths) const {
    return impl_->applyA(mu, smoothingWidths);
}

template <typename Scalar>
std::vector<Vec3<Scalar>> WindingNumberOperator<Scalar>::applyATransposed(std::span<const Scalar> values, std::span<const Scalar> smoothingWidths) const {
    return impl_->applyATransposed(values, smoothingWidths);
}

template <typename Scalar>
std::vector<Vec3<Scalar>> WindingNumberOperator<Scalar>::applyG(std::span<const Vec3<Scalar>> mu, std::span<const Scalar> smoothingWidths) const {
    return impl_->applyG(mu, smoothingWidths);
}

// Explicit instantiations for double and float
template class WindingNumberOperator<double>;
template class WindingNumberOperator<float>;

} // namespace wnnc
