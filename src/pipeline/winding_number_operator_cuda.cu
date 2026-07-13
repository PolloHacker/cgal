// SPDX-License-Identifier: MIT
#include "wnnc/winding_number_operator.h"

#include <cuda_runtime.h>
#include "wn_treecode_cpu.h"
#include "wn_treecode_cuda.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace wnnc {

template <typename Scalar>
struct WindingNumberOperator<Scalar>::Impl {
    Index point_count = 0;
    std::vector<Vec3<Scalar>> points_host;

    // Tree dimensions
    signedindex_t num_nodes = 0;
    signedindex_t num_leaves = 0;
    signedindex_t tree_depth = 0;

    // GPU buffers for the tree (all single precision float)
    signedindex_t* node_parent_list_d = nullptr;
    signedindex_t* node_children_list_d = nullptr;
    bool* node_is_leaf_list_d = nullptr;
    signedindex_t* num_points_in_node_d = nullptr;
    signedindex_t* node2point_indexstart_d = nullptr;
    float* node_half_w_list_d = nullptr;
    signedindex_t* node2point_index_d = nullptr;
    float* points_d = nullptr;

    // Pre-allocated temporary/helper GPU buffers to eliminate allocator overhead
    float* temp_mu_d = nullptr;                  // point_count * 3 * sizeof(float)
    float* temp_weights_d = nullptr;             // point_count * sizeof(float)
    float* temp_widths_d = nullptr;              // point_count * sizeof(float)
    bool* temp_scattered_mask_d = nullptr;       // num_nodes * sizeof(bool)
    bool* temp_next_to_scatter_mask_d = nullptr;  // num_nodes * sizeof(bool)
    float* temp_node_attrs_d = nullptr;          // num_nodes * 3 * sizeof(float)
    float* temp_node_reppoints_d = nullptr;      // num_nodes * 3 * sizeof(float)
    float* temp_node_weights_d = nullptr;        // num_nodes * sizeof(float)
    float* temp_out_attrs_d = nullptr;           // point_count * 3 * sizeof(float)

    // Pre-allocated CPU vectors to eliminate vector reallocation overhead
    mutable std::vector<float> host_flat_mu;
    mutable std::vector<float> host_weights;
    mutable std::vector<float> host_widths;

    Impl(std::vector<Vec3<Scalar>> points, int maxTreeDepth, Index maxPointsPerLeaf)
        : point_count(static_cast<Index>(points.size())), points_host(std::move(points)) {
        if (point_count == 0) {
            throw std::invalid_argument("WindingNumberOperator: empty point set");
        }

        // 1. Flatten and cast coordinates to float
        std::vector<float> flat_points(point_count * 3);
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
        std::vector<signedindex_t> node_parent_list(num_nodes);
        std::vector<signedindex_t> node_children_list(num_nodes * NUM_OCT_CHILDREN);
        std::unique_ptr<bool[]> node_is_leaf_list(new bool[num_nodes]);
        std::vector<float> node_half_w_list(num_nodes);
        std::vector<signedindex_t> num_points_in_node(num_nodes);
        std::vector<signedindex_t> node2point_indexstart(num_nodes);
        std::vector<signedindex_t> stdvec_node2point_index;

        serialize_tree_recursive<float>(
            root,
            node_parent_list.data(),
            node_children_list.data(),
            node_is_leaf_list.get(),
            node_half_w_list.data(),
            num_points_in_node.data(),
            node2point_indexstart.data(),
            stdvec_node2point_index
        );

        free_tree_recursive<float>(root);

        // 4. Allocate GPU buffers
        cudaMalloc(&node_parent_list_d, num_nodes * sizeof(signedindex_t));
        cudaMalloc(&node_children_list_d, num_nodes * NUM_OCT_CHILDREN * sizeof(signedindex_t));
        cudaMalloc(&node_is_leaf_list_d, num_nodes * sizeof(bool));
        cudaMalloc(&num_points_in_node_d, num_nodes * sizeof(signedindex_t));
        cudaMalloc(&node2point_indexstart_d, num_nodes * sizeof(signedindex_t));
        cudaMalloc(&node_half_w_list_d, num_nodes * sizeof(float));
        cudaMalloc(&node2point_index_d, stdvec_node2point_index.size() * sizeof(signedindex_t));
        cudaMalloc(&points_d, point_count * 3 * sizeof(float));

        // Pre-allocate temporary/helper buffers on GPU
        cudaMalloc(&temp_mu_d, point_count * 3 * sizeof(float));
        cudaMalloc(&temp_weights_d, point_count * sizeof(float));
        cudaMalloc(&temp_widths_d, point_count * sizeof(float));
        cudaMalloc(&temp_scattered_mask_d, num_nodes * sizeof(bool));
        cudaMalloc(&temp_next_to_scatter_mask_d, num_nodes * sizeof(bool));
        cudaMalloc(&temp_node_attrs_d, num_nodes * 3 * sizeof(float));
        cudaMalloc(&temp_node_reppoints_d, num_nodes * 3 * sizeof(float));
        cudaMalloc(&temp_node_weights_d, num_nodes * sizeof(float));
        cudaMalloc(&temp_out_attrs_d, point_count * 3 * sizeof(float));

        // Resize mutable host buffers
        host_flat_mu.resize(point_count * 3);
        host_weights.resize(point_count);
        host_widths.resize(point_count);

        // 5. Copy to GPU
        cudaMemcpy(node_parent_list_d, node_parent_list.data(), num_nodes * sizeof(signedindex_t), cudaMemcpyHostToDevice);
        cudaMemcpy(node_children_list_d, node_children_list.data(), num_nodes * NUM_OCT_CHILDREN * sizeof(signedindex_t), cudaMemcpyHostToDevice);
        cudaMemcpy(node_is_leaf_list_d, node_is_leaf_list.get(), num_nodes * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(num_points_in_node_d, num_points_in_node.data(), num_nodes * sizeof(signedindex_t), cudaMemcpyHostToDevice);
        cudaMemcpy(node2point_indexstart_d, node2point_indexstart.data(), num_nodes * sizeof(signedindex_t), cudaMemcpyHostToDevice);
        cudaMemcpy(node_half_w_list_d, node_half_w_list.data(), num_nodes * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(node2point_index_d, stdvec_node2point_index.data(), stdvec_node2point_index.size() * sizeof(signedindex_t), cudaMemcpyHostToDevice);
        cudaMemcpy(points_d, flat_points.data(), point_count * 3 * sizeof(float), cudaMemcpyHostToDevice);
    }

    ~Impl() {
        cudaFree(node_parent_list_d);
        cudaFree(node_children_list_d);
        cudaFree(node_is_leaf_list_d);
        cudaFree(num_points_in_node_d);
        cudaFree(node2point_indexstart_d);
        cudaFree(node_half_w_list_d);
        cudaFree(node2point_index_d);
        cudaFree(points_d);

        cudaFree(temp_mu_d);
        cudaFree(temp_weights_d);
        cudaFree(temp_widths_d);
        cudaFree(temp_scattered_mask_d);
        cudaFree(temp_next_to_scatter_mask_d);
        cudaFree(temp_node_attrs_d);
        cudaFree(temp_node_reppoints_d);
        cudaFree(temp_node_weights_d);
        cudaFree(temp_out_attrs_d);
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

        cudaMemcpy(temp_mu_d, host_flat_mu.data(), point_count * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_weights_d, host_weights.data(), point_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_widths_d, host_widths.data(), point_count * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(temp_scattered_mask_d, 0, num_nodes * sizeof(bool));
        cudaMemset(temp_next_to_scatter_mask_d, 0, num_nodes * sizeof(bool));
        cudaMemset(temp_node_attrs_d, 0, num_nodes * 3 * sizeof(float));
        cudaMemset(temp_node_reppoints_d, 0, num_nodes * 3 * sizeof(float));
        cudaMemset(temp_node_weights_d, 0, num_nodes * sizeof(float));

        signedindex_t num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        scatter_point_attrs_to_nodes_leaf_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
            node_parent_list_d,
            points_d,
            temp_weights_d,
            temp_mu_d,
            node2point_index_d,
            node2point_indexstart_d,
            num_points_in_node_d,
            node_is_leaf_list_d,
            temp_scattered_mask_d,
            temp_node_attrs_d,
            temp_node_reppoints_d,
            temp_node_weights_d,
            3, // attr_dim = 3
            num_nodes
        );

        for (signedindex_t depth = tree_depth - 1; depth >= 0; depth--) {
            find_next_to_scatter_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
                node_children_list_d,
                node_is_leaf_list_d,
                temp_scattered_mask_d,
                temp_next_to_scatter_mask_d,
                node2point_index_d,
                num_nodes
            );

            scatter_point_attrs_to_nodes_nonleaf_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
                node_parent_list_d,
                node_children_list_d,
                points_d,
                temp_weights_d,
                temp_mu_d,
                node2point_index_d,
                node2point_indexstart_d,
                num_points_in_node_d,
                node_is_leaf_list_d,
                temp_scattered_mask_d,
                temp_next_to_scatter_mask_d,
                temp_node_attrs_d,
                temp_node_reppoints_d,
                temp_node_weights_d,
                3, // attr_dim = 3
                num_nodes
            );
        }

        // Multiply by A
        cudaMemset(temp_out_attrs_d, 0, point_count * sizeof(float));

        signedindex_t num_blocks_queries = (point_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        multiply_by_A_cuda_kernel<float><<<num_blocks_queries, THREADS_PER_BLOCK>>>(
            points_d,
            temp_widths_d,
            points_d,
            temp_mu_d,
            node2point_index_d,
            node2point_indexstart_d,
            node_children_list_d,
            temp_node_attrs_d,
            node_is_leaf_list_d,
            node_half_w_list_d,
            temp_node_reppoints_d,
            num_points_in_node_d,
            temp_out_attrs_d,
            point_count,
            true
        );

        std::vector<float> out_attrs_h(point_count);
        cudaMemcpy(out_attrs_h.data(), temp_out_attrs_d, point_count * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<Scalar> result(point_count);
        for (size_t i = 0; i < (size_t)point_count; ++i) {
            result[i] = static_cast<Scalar>(out_attrs_h[i]);
        }
        return result;
    }

    std::vector<Vec3<Scalar>> applyATransposed(std::span<const Scalar> values, std::span<const Scalar> smoothingWidths) const {
        for (size_t i = 0; i < (size_t)point_count; ++i) {
            float val = static_cast<float>(values[i]);
            host_flat_mu[i] = val;
            host_weights[i] = std::abs(val);
            host_widths[i] = static_cast<float>(smoothingWidths[i]);
        }

        cudaMemcpy(temp_mu_d, host_flat_mu.data(), point_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_weights_d, host_weights.data(), point_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_widths_d, host_widths.data(), point_count * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(temp_scattered_mask_d, 0, num_nodes * sizeof(bool));
        cudaMemset(temp_next_to_scatter_mask_d, 0, num_nodes * sizeof(bool));
        cudaMemset(temp_node_attrs_d, 0, num_nodes * sizeof(float));
        cudaMemset(temp_node_reppoints_d, 0, num_nodes * 3 * sizeof(float));
        cudaMemset(temp_node_weights_d, 0, num_nodes * sizeof(float));

        signedindex_t num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        scatter_point_attrs_to_nodes_leaf_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
            node_parent_list_d,
            points_d,
            temp_weights_d,
            temp_mu_d,
            node2point_index_d,
            node2point_indexstart_d,
            num_points_in_node_d,
            node_is_leaf_list_d,
            temp_scattered_mask_d,
            temp_node_attrs_d,
            temp_node_reppoints_d,
            temp_node_weights_d,
            1, // attr_dim = 1
            num_nodes
        );

        for (signedindex_t depth = tree_depth - 1; depth >= 0; depth--) {
            find_next_to_scatter_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
                node_children_list_d,
                node_is_leaf_list_d,
                temp_scattered_mask_d,
                temp_next_to_scatter_mask_d,
                node2point_index_d,
                num_nodes
            );

            scatter_point_attrs_to_nodes_nonleaf_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
                node_parent_list_d,
                node_children_list_d,
                points_d,
                temp_weights_d,
                temp_mu_d,
                node2point_index_d,
                node2point_indexstart_d,
                num_points_in_node_d,
                node_is_leaf_list_d,
                temp_scattered_mask_d,
                temp_next_to_scatter_mask_d,
                temp_node_attrs_d,
                temp_node_reppoints_d,
                temp_node_weights_d,
                1, // attr_dim = 1
                num_nodes
            );
        }

        // Multiply by AT
        cudaMemset(temp_out_attrs_d, 0, point_count * 3 * sizeof(float));

        signedindex_t num_blocks_queries = (point_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        multiply_by_AT_cuda_kernel<float><<<num_blocks_queries, THREADS_PER_BLOCK>>>(
            points_d,
            temp_widths_d,
            points_d,
            temp_mu_d,
            node2point_index_d,
            node2point_indexstart_d,
            node_children_list_d,
            temp_node_attrs_d,
            node_is_leaf_list_d,
            node_half_w_list_d,
            temp_node_reppoints_d,
            num_points_in_node_d,
            temp_out_attrs_d,
            point_count
        );

        std::vector<float> out_attrs_h(point_count * 3);
        cudaMemcpy(out_attrs_h.data(), temp_out_attrs_d, point_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<Vec3<Scalar>> result(point_count);
        for (size_t i = 0; i < (size_t)point_count; ++i) {
            result[i] = {
                static_cast<Scalar>(out_attrs_h[3 * i + 0]),
                static_cast<Scalar>(out_attrs_h[3 * i + 1]),
                static_cast<Scalar>(out_attrs_h[3 * i + 2])
            };
        }
        return result;
    }

    std::vector<Vec3<Scalar>> applyG(std::span<const Vec3<Scalar>> mu, std::span<const Scalar> smoothingWidths) const {
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

        cudaMemcpy(temp_mu_d, host_flat_mu.data(), point_count * 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_weights_d, host_weights.data(), point_count * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(temp_widths_d, host_widths.data(), point_count * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(temp_scattered_mask_d, 0, num_nodes * sizeof(bool));
        cudaMemset(temp_next_to_scatter_mask_d, 0, num_nodes * sizeof(bool));
        cudaMemset(temp_node_attrs_d, 0, num_nodes * 3 * sizeof(float));
        cudaMemset(temp_node_reppoints_d, 0, num_nodes * 3 * sizeof(float));
        cudaMemset(temp_node_weights_d, 0, num_nodes * sizeof(float));

        signedindex_t num_blocks = (num_nodes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        scatter_point_attrs_to_nodes_leaf_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
            node_parent_list_d,
            points_d,
            temp_weights_d,
            temp_mu_d,
            node2point_index_d,
            node2point_indexstart_d,
            num_points_in_node_d,
            node_is_leaf_list_d,
            temp_scattered_mask_d,
            temp_node_attrs_d,
            temp_node_reppoints_d,
            temp_node_weights_d,
            3, // attr_dim = 3
            num_nodes
        );

        for (signedindex_t depth = tree_depth - 1; depth >= 0; depth--) {
            find_next_to_scatter_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
                node_children_list_d,
                node_is_leaf_list_d,
                temp_scattered_mask_d,
                temp_next_to_scatter_mask_d,
                node2point_index_d,
                num_nodes
            );

            scatter_point_attrs_to_nodes_nonleaf_cuda_kernel<float><<<num_blocks, THREADS_PER_BLOCK>>>(
                node_parent_list_d,
                node_children_list_d,
                points_d,
                temp_weights_d,
                temp_mu_d,
                node2point_index_d,
                node2point_indexstart_d,
                num_points_in_node_d,
                node_is_leaf_list_d,
                temp_scattered_mask_d,
                temp_next_to_scatter_mask_d,
                temp_node_attrs_d,
                temp_node_reppoints_d,
                temp_node_weights_d,
                3, // attr_dim = 3
                num_nodes
            );
        }

        // Multiply by G
        cudaMemset(temp_out_attrs_d, 0, point_count * 3 * sizeof(float));

        signedindex_t num_blocks_queries = (point_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        multiply_by_G_cuda_kernel<float><<<num_blocks_queries, THREADS_PER_BLOCK>>>(
            points_d,
            temp_widths_d,
            points_d,
            temp_mu_d,
            node2point_index_d,
            node2point_indexstart_d,
            node_children_list_d,
            temp_node_attrs_d,
            node_is_leaf_list_d,
            node_half_w_list_d,
            temp_node_reppoints_d,
            num_points_in_node_d,
            temp_out_attrs_d,
            point_count
        );

        std::vector<float> out_attrs_h(point_count * 3);
        cudaMemcpy(out_attrs_h.data(), temp_out_attrs_d, point_count * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<Vec3<Scalar>> result(point_count);
        for (size_t i = 0; i < (size_t)point_count; ++i) {
            result[i] = {
                static_cast<Scalar>(out_attrs_h[3 * i + 0]),
                static_cast<Scalar>(out_attrs_h[3 * i + 1]),
                static_cast<Scalar>(out_attrs_h[3 * i + 2])
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
