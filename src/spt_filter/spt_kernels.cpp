// SPDX-License-Identifier: MIT
// High-performance OpenMP + Eigen3 C++ kernels for Multi-Scale Geometric Features,
// 3D Cone Enclosure Raycasting, and l0-Cut Pursuit Superpoint Partitioning.

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <omp.h>
#include <queue>
#include <unordered_map>
#include <vector>

#include "wnnc/normal_solver.h"
#include "wnnc/vec3.h"
#include "wnnc/winding_number_operator.h"

namespace {

struct GridIndex {
    int32_t ix, iy, iz;
    bool operator==(const GridIndex& o) const noexcept {
        return ix == o.ix && iy == o.iy && iz == o.iz;
    }
};

inline uint64_t hash_coords(int32_t ix, int32_t iy, int32_t iz) noexcept {
    // 64-bit spatial hash (Teschner et al.)
    const uint64_t ux = static_cast<uint32_t>(ix);
    const uint64_t uy = static_cast<uint32_t>(iy);
    const uint64_t uz = static_cast<uint32_t>(iz);
    return (ux * 73856093ULL) ^ (uy * 19349663ULL) ^ (uz * 83492791ULL);
}

struct SpatialGrid {
    float inv_cell_size;
    float min_x, min_y, min_z;
    uint32_t table_mask;
    std::vector<int32_t> bucket_head;
    std::vector<int32_t> next_node;
    std::vector<int32_t> cell_ix, cell_iy, cell_iz;
    std::vector<int32_t> point_start;
    std::vector<int32_t> point_count;
    std::vector<int32_t> sorted_points;

    void build(const float* xyz, int32_t n_pts, float cell_size) {
        inv_cell_size = 1.0f / cell_size;
        min_x = xyz[0];
        min_y = xyz[1];
        min_z = xyz[2];
        for (int32_t i = 1; i < n_pts; ++i) {
            min_x = std::min(min_x, xyz[3 * i + 0]);
            min_y = std::min(min_y, xyz[3 * i + 1]);
            min_z = std::min(min_z, xyz[3 * i + 2]);
        }

        uint32_t cap = 1;
        while (cap < static_cast<uint32_t>(n_pts) * 2U) cap <<= 1U;
        table_mask = cap - 1U;
        bucket_head.assign(cap, -1);

        std::vector<int32_t> pt_cell_id(n_pts, -1);
        int32_t num_cells = 0;

        cell_ix.clear();
        cell_iy.clear();
        cell_iz.clear();
        next_node.clear();
        point_count.clear();

        cell_ix.reserve(n_pts / 2 + 16);
        cell_iy.reserve(n_pts / 2 + 16);
        cell_iz.reserve(n_pts / 2 + 16);
        next_node.reserve(n_pts / 2 + 16);
        point_count.reserve(n_pts / 2 + 16);

        for (int32_t i = 0; i < n_pts; ++i) {
            int32_t ix = static_cast<int32_t>(std::floor((xyz[3 * i + 0] - min_x) * inv_cell_size));
            int32_t iy = static_cast<int32_t>(std::floor((xyz[3 * i + 1] - min_y) * inv_cell_size));
            int32_t iz = static_cast<int32_t>(std::floor((xyz[3 * i + 2] - min_z) * inv_cell_size));
            uint32_t b = static_cast<uint32_t>(hash_coords(ix, iy, iz)) & table_mask;

            int32_t cid = bucket_head[b];
            while (cid != -1) {
                if (cell_ix[cid] == ix && cell_iy[cid] == iy && cell_iz[cid] == iz) {
                    break;
                }
                cid = next_node[cid];
            }
            if (cid == -1) {
                cid = num_cells++;
                cell_ix.push_back(ix);
                cell_iy.push_back(iy);
                cell_iz.push_back(iz);
                next_node.push_back(bucket_head[b]);
                bucket_head[b] = cid;
                point_count.push_back(0);
            }
            pt_cell_id[i] = cid;
            point_count[cid]++;
        }

        point_start.resize(num_cells + 1, 0);
        for (int32_t c = 0; c < num_cells; ++c) {
            point_start[c + 1] = point_start[c] + point_count[c];
        }
        sorted_points.resize(n_pts);
        std::vector<int32_t> write_offset = point_start;
        for (int32_t i = 0; i < n_pts; ++i) {
            int32_t cid = pt_cell_id[i];
            sorted_points[write_offset[cid]++] = i;
        }
    }

    inline int32_t find_cell(int32_t ix, int32_t iy, int32_t iz) const noexcept {
        uint32_t b = static_cast<uint32_t>(hash_coords(ix, iy, iz)) & table_mask;
        int32_t cid = bucket_head[b];
        while (cid != -1) {
            if (cell_ix[cid] == ix && cell_iy[cid] == iy && cell_iz[cid] == iz) {
                return cid;
            }
            cid = next_node[cid];
        }
        return -1;
    }

    inline void point_to_cell(float x, float y, float z, int32_t& ix, int32_t& iy, int32_t& iz) const noexcept {
        ix = static_cast<int32_t>(std::floor((x - min_x) * inv_cell_size));
        iy = static_cast<int32_t>(std::floor((y - min_y) * inv_cell_size));
        iz = static_cast<int32_t>(std::floor((z - min_z) * inv_cell_size));
    }
};

} // namespace

extern "C" {

// -----------------------------------------------------------------------------
// 1. Voxel Grid Subsampling (P0)
// Returns number of P0 voxels M, fills p0_to_orig[M] (representative point index)
// and orig_to_p0[N] (voxel index for each original point).
// -----------------------------------------------------------------------------
int32_t spt_voxel_subsample(
    const float* xyz,
    int32_t n_pts,
    float voxel_size,
    int32_t* p0_to_orig,
    int32_t* orig_to_p0
) {
    if (n_pts <= 0) return 0;
    SpatialGrid grid;
    grid.build(xyz, n_pts, voxel_size);
    const int32_t num_voxels = static_cast<int32_t>(grid.point_count.size());

    #pragma omp parallel for schedule(static)
    for (int32_t c = 0; c < num_voxels; ++c) {
        const int32_t start = grid.point_start[c];
        const int32_t count = grid.point_count[c];
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (int32_t k = 0; k < count; ++k) {
            int32_t idx = grid.sorted_points[start + k];
            orig_to_p0[idx] = c;
            cx += xyz[3 * idx + 0];
            cy += xyz[3 * idx + 1];
            cz += xyz[3 * idx + 2];
        }
        cx /= count;
        cy /= count;
        cz /= count;

        int32_t best_idx = grid.sorted_points[start];
        double best_d2 = 1e30;
        for (int32_t k = 0; k < count; ++k) {
            int32_t idx = grid.sorted_points[start + k];
            double dx = xyz[3 * idx + 0] - cx;
            double dy = xyz[3 * idx + 1] - cy;
            double dz = xyz[3 * idx + 2] - cz;
            double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = idx;
            }
        }
        p0_to_orig[c] = best_idx;
    }
    return num_voxels;
}

// -----------------------------------------------------------------------------
// 2. Multi-Scale Geometric Feature Extraction & Normal Estimation
// Outputs per-point features at a single scale r:
//   feat_out[i * 6 + 0] = rugosity (surface variation: l3 / (l1+l2+l3))
//   feat_out[i * 6 + 1] = planarity ((l2 - l3) / l1)
//   feat_out[i * 6 + 2] = scattering (sphericity: l3 / l1)
//   feat_out[i * 6 + 3] = linearity ((l1 - l2) / l1)
//   feat_out[i * 6 + 4] = omnivariance (cbrt(l1*l2*l3) / (l1+l2+l3))
//   feat_out[i * 6 + 5] = normal_dispersion (1 - ||mean(n_j)||)
// If compute_normals != 0, also writes estimated PCA normals to normals_inout[n_pts * 3]
// oriented consistently toward the local tunnel interior centroid.
// -----------------------------------------------------------------------------
void spt_compute_scale_features(
    const float* xyz,
    float* normals_inout,
    int32_t n_pts,
    float radius,
    int32_t compute_normals,
    float* feat_out
) {
    // For coarse scales (radius >= 0.30m) on large clouds when normals are already known,
    // subsample a uniform carrier grid (cell_size = radius * 0.22f, e.g. 11cm for r=0.50m)
    // so each 0.50m sphere contains ~60 uniform carrier points instead of 500+ dense P0 points.
    std::vector<float> carrier_xyz;
    std::vector<float> carrier_normals;
    const float* search_xyz = xyz;
    const float* search_normals = normals_inout;
    int32_t n_search = n_pts;

    if (compute_normals == 0 && radius >= 0.30f && n_pts > 50000) {
        const float carrier_cell = radius * 0.22f;
        SpatialGrid cgrid;
        cgrid.build(xyz, n_pts, carrier_cell);
        const int32_t n_carr = static_cast<int32_t>(cgrid.point_count.size());
        carrier_xyz.resize(static_cast<size_t>(n_carr) * 3);
        carrier_normals.resize(static_cast<size_t>(n_carr) * 3);

        #pragma omp parallel for schedule(static)
        for (int32_t c = 0; c < n_carr; ++c) {
            const int32_t start = cgrid.point_start[c];
            const int32_t count = cgrid.point_count[c];
            double cx = 0.0, cy = 0.0, cz = 0.0;
            for (int32_t k = 0; k < count; ++k) {
                int32_t idx = cgrid.sorted_points[start + k];
                cx += xyz[3 * idx + 0];
                cy += xyz[3 * idx + 1];
                cz += xyz[3 * idx + 2];
            }
            cx /= count;
            cy /= count;
            cz /= count;
            int32_t rep = cgrid.sorted_points[start];
            carrier_xyz[3 * c + 0] = static_cast<float>(cx);
            carrier_xyz[3 * c + 1] = static_cast<float>(cy);
            carrier_xyz[3 * c + 2] = static_cast<float>(cz);
            carrier_normals[3 * c + 0] = normals_inout[3 * rep + 0];
            carrier_normals[3 * c + 1] = normals_inout[3 * rep + 1];
            carrier_normals[3 * c + 2] = normals_inout[3 * rep + 2];
        }
        search_xyz = carrier_xyz.data();
        search_normals = carrier_normals.data();
        n_search = n_carr;
    }

    SpatialGrid grid;
    grid.build(search_xyz, n_search, radius);
    const float r2 = radius * radius;
    const float r2_expand = (2.0f * radius) * (2.0f * radius);
    const float r2_expand3 = (3.0f * radius) * (3.0f * radius);

    // Step A: When compute_normals != 0, compute local PCA eigenvalues and eigenvectors,
    // and orient/blend with the global WNNC normals already provided in normals_inout.
    // We also keep pure local PCA directions in pca_dirs for sharp normal_dispersion computation.
    std::vector<float> pca_dirs;
    if (compute_normals != 0) {
        pca_dirs.resize(static_cast<size_t>(n_pts) * 3);

        #pragma omp parallel for schedule(dynamic, 1024)
        for (int32_t i = 0; i < n_pts; ++i) {
            const float px = xyz[3 * i + 0];
            const float py = xyz[3 * i + 1];
            const float pz = xyz[3 * i + 2];
            int32_t ix, iy, iz;
            grid.point_to_cell(px, py, pz, ix, iy, iz);

            double mx = 0.0, my = 0.0, mz = 0.0;
            double cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;
            int32_t cnt = 0;

            for (int32_t dz = -1; dz <= 1; ++dz) {
                for (int32_t dy = -1; dy <= 1; ++dy) {
                    for (int32_t dx = -1; dx <= 1; ++dx) {
                        int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                        if (cid == -1) continue;
                        const int32_t start = grid.point_start[cid];
                        const int32_t end = start + grid.point_count[cid];
                        for (int32_t k = start; k < end; ++k) {
                            int32_t j = grid.sorted_points[k];
                            float qx = search_xyz[3 * j + 0] - px;
                            float qy = search_xyz[3 * j + 1] - py;
                            float qz = search_xyz[3 * j + 2] - pz;
                            if (qx * qx + qy * qy + qz * qz <= r2) {
                                mx += qx; my += qy; mz += qz;
                                cxx += qx * qx; cxy += qx * qy; cxz += qx * qz;
                                cyy += qy * qy; cyz += qy * qz; czz += qz * qz;
                                cnt++;
                            }
                        }
                    }
                }
            }

            // If point is in a low-density region (< 6 neighbors within r), expand search to 2*r and 3*r
            if (cnt < 6) {
                mx = my = mz = 0.0;
                cxx = cxy = cxz = cyy = cyz = czz = 0.0;
                cnt = 0;
                for (int32_t dz = -2; dz <= 2; ++dz) {
                    for (int32_t dy = -2; dy <= 2; ++dy) {
                        for (int32_t dx = -2; dx <= 2; ++dx) {
                            int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                            if (cid == -1) continue;
                            const int32_t start = grid.point_start[cid];
                            const int32_t end = start + grid.point_count[cid];
                            for (int32_t k = start; k < end; ++k) {
                                int32_t j = grid.sorted_points[k];
                                float qx = search_xyz[3 * j + 0] - px;
                                float qy = search_xyz[3 * j + 1] - py;
                                float qz = search_xyz[3 * j + 2] - pz;
                                if (qx * qx + qy * qy + qz * qz <= r2_expand) {
                                    mx += qx; my += qy; mz += qz;
                                    cxx += qx * qx; cxy += qx * qy; cxz += qx * qz;
                                    cyy += qy * qy; cyz += qy * qz; czz += qz * qz;
                                    cnt++;
                                }
                            }
                        }
                    }
                }
            }
            if (cnt < 6) {
                mx = my = mz = 0.0;
                cxx = cxy = cxz = cyy = cyz = czz = 0.0;
                cnt = 0;
                for (int32_t dz = -3; dz <= 3; ++dz) {
                    for (int32_t dy = -3; dy <= 3; ++dy) {
                        for (int32_t dx = -3; dx <= 3; ++dx) {
                            int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                            if (cid == -1) continue;
                            const int32_t start = grid.point_start[cid];
                            const int32_t end = start + grid.point_count[cid];
                            for (int32_t k = start; k < end; ++k) {
                                int32_t j = grid.sorted_points[k];
                                float qx = search_xyz[3 * j + 0] - px;
                                float qy = search_xyz[3 * j + 1] - py;
                                float qz = search_xyz[3 * j + 2] - pz;
                                if (qx * qx + qy * qy + qz * qz <= r2_expand3) {
                                    mx += qx; my += qy; mz += qz;
                                    cxx += qx * qx; cxy += qx * qy; cxz += qx * qz;
                                    cyy += qy * qy; cyz += qy * qz; czz += qz * qz;
                                    cnt++;
                                }
                            }
                        }
                    }
                }
            }

            Eigen::Vector3f wnnc_n(normals_inout[3 * i + 0], normals_inout[3 * i + 1], normals_inout[3 * i + 2]);
            float wnnc_norm = wnnc_n.norm();
            if (wnnc_norm > 1e-6f) {
                wnnc_n /= wnnc_norm;
            } else {
                wnnc_n = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            }

            Eigen::Vector3f n_vec = wnnc_n;
            if (cnt >= 4) {
                double inv_n = 1.0 / cnt;
                mx *= inv_n; my *= inv_n; mz *= inv_n;
                Eigen::Matrix3f cov;
                cov(0, 0) = static_cast<float>(cxx * inv_n - mx * mx);
                cov(0, 1) = cov(1, 0) = static_cast<float>(cxy * inv_n - mx * my);
                cov(0, 2) = cov(2, 0) = static_cast<float>(cxz * inv_n - mx * mz);
                cov(1, 1) = static_cast<float>(cyy * inv_n - my * my);
                cov(1, 2) = cov(2, 1) = static_cast<float>(cyz * inv_n - my * mz);
                cov(2, 2) = static_cast<float>(czz * inv_n - mz * mz);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
                es.computeDirect(cov);
                n_vec = es.eigenvectors().col(0);
                if (n_vec.dot(wnnc_n) < 0.0f) {
                    n_vec = -n_vec;
                }
                const float l3 = std::max(0.0f, es.eigenvalues()(0));
                const float l2 = std::max(l3, es.eigenvalues()(1));
                const float l1 = std::max(l2, es.eigenvalues()(2));
                const float sum_l = l1 + l2 + l3 + 1e-12f;
                const float l1_eps = l1 + 1e-12f;
                feat_out[6 * i + 0] = l3 / sum_l;
                feat_out[6 * i + 1] = (l2 - l3) / l1_eps;
                feat_out[6 * i + 2] = l3 / l1_eps;
                feat_out[6 * i + 3] = (l1 - l2) / l1_eps;
                feat_out[6 * i + 4] = std::cbrt(l1 * l2 * l3) / sum_l;
                feat_out[6 * i + 5] = 0.0f;
            } else {
                // Neutral fallback for truly isolated points (neither high-entropy canopy nor solid rock roof)
                feat_out[6 * i + 0] = 0.11f;
                feat_out[6 * i + 1] = 0.30f;
                feat_out[6 * i + 2] = 0.22f;
                feat_out[6 * i + 3] = 0.20f;
                feat_out[6 * i + 4] = 0.18f;
                feat_out[6 * i + 5] = 0.32f;
            }
            pca_dirs[3 * i + 0] = n_vec.x();
            pca_dirs[3 * i + 1] = n_vec.y();
            pca_dirs[3 * i + 2] = n_vec.z();

            // Combine global WNNC orientation with local PCA normal on planar surfaces,
            // while keeping unsmoothed local PCA normal on non-planar 3D scattering structures
            float w_wnnc = 0.60f * std::clamp((feat_out[6 * i + 1] - 0.25f) * 2.5f, 0.0f, 1.0f);
            Eigen::Vector3f blended = w_wnnc * wnnc_n + (1.0f - w_wnnc) * n_vec;
            float b_norm = blended.norm();
            if (b_norm > 1e-6f) blended /= b_norm;
            else blended = wnnc_n;

            normals_inout[3 * i + 0] = blended.x();
            normals_inout[3 * i + 1] = blended.y();
            normals_inout[3 * i + 2] = blended.z();
        }
        search_normals = pca_dirs.data();
    }

    // Step B: Compute eigenvalues (if compute_normals == 0) and sign-invariant normal structure tensor dispersion
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int32_t i = 0; i < n_pts; ++i) {
        const float px = xyz[3 * i + 0];
        const float py = xyz[3 * i + 1];
        const float pz = xyz[3 * i + 2];
        int32_t ix, iy, iz;
        grid.point_to_cell(px, py, pz, ix, iy, iz);

        double mx = 0.0, my = 0.0, mz = 0.0;
        double cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;
        // Sign-invariant normal structure tensor T = (1/m) sum(n_j * n_j^T)
        double nxx = 0.0, nxy = 0.0, nxz = 0.0, nyy = 0.0, nyz = 0.0, nzz = 0.0;
        int32_t cnt = 0;

        for (int32_t dz = -1; dz <= 1; ++dz) {
            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                    if (cid == -1) continue;
                    const int32_t start = grid.point_start[cid];
                    const int32_t end = start + grid.point_count[cid];
                    for (int32_t k = start; k < end; ++k) {
                        int32_t j = grid.sorted_points[k];
                        float qx = search_xyz[3 * j + 0] - px;
                        float qy = search_xyz[3 * j + 1] - py;
                        float qz = search_xyz[3 * j + 2] - pz;
                        if (qx * qx + qy * qy + qz * qz <= r2) {
                            if (compute_normals == 0) {
                                mx += qx; my += qy; mz += qz;
                                cxx += qx * qx; cxy += qx * qy; cxz += qx * qz;
                                cyy += qy * qy; cyz += qy * qz; czz += qz * qz;
                            }
                            float njx = search_normals[3 * j + 0];
                            float njy = search_normals[3 * j + 1];
                            float njz = search_normals[3 * j + 2];
                            nxx += njx * njx; nxy += njx * njy; nxz += njx * njz;
                            nyy += njy * njy; nyz += njy * njz; nzz += njz * njz;
                            cnt++;
                        }
                    }
                }
            }
        }

        if (cnt < 6) {
            mx = my = mz = 0.0;
            cxx = cxy = cxz = cyy = cyz = czz = 0.0;
            nxx = nxy = nxz = nyy = nyz = nzz = 0.0;
            cnt = 0;
            for (int32_t dz = -2; dz <= 2; ++dz) {
                for (int32_t dy = -2; dy <= 2; ++dy) {
                    for (int32_t dx = -2; dx <= 2; ++dx) {
                        int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                        if (cid == -1) continue;
                        const int32_t start = grid.point_start[cid];
                        const int32_t end = start + grid.point_count[cid];
                        for (int32_t k = start; k < end; ++k) {
                            int32_t j = grid.sorted_points[k];
                            float qx = search_xyz[3 * j + 0] - px;
                            float qy = search_xyz[3 * j + 1] - py;
                            float qz = search_xyz[3 * j + 2] - pz;
                            if (qx * qx + qy * qy + qz * qz <= r2_expand) {
                                if (compute_normals == 0) {
                                    mx += qx; my += qy; mz += qz;
                                    cxx += qx * qx; cxy += qx * qy; cxz += qx * qz;
                                    cyy += qy * qy; cyz += qy * qz; czz += qz * qz;
                                }
                                float njx = search_normals[3 * j + 0];
                                float njy = search_normals[3 * j + 1];
                                float njz = search_normals[3 * j + 2];
                                nxx += njx * njx; nxy += njx * njy; nxz += njx * njz;
                                nyy += njy * njy; nyz += njy * njz; nzz += njz * njz;
                                cnt++;
                            }
                        }
                    }
                }
            }
        }
        if (cnt < 6) {
            mx = my = mz = 0.0;
            cxx = cxy = cxz = cyy = cyz = czz = 0.0;
            nxx = nxy = nxz = nyy = nyz = nzz = 0.0;
            cnt = 0;
            for (int32_t dz = -3; dz <= 3; ++dz) {
                for (int32_t dy = -3; dy <= 3; ++dy) {
                    for (int32_t dx = -3; dx <= 3; ++dx) {
                        int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                        if (cid == -1) continue;
                        const int32_t start = grid.point_start[cid];
                        const int32_t end = start + grid.point_count[cid];
                        for (int32_t k = start; k < end; ++k) {
                            int32_t j = grid.sorted_points[k];
                            float qx = search_xyz[3 * j + 0] - px;
                            float qy = search_xyz[3 * j + 1] - py;
                            float qz = search_xyz[3 * j + 2] - pz;
                            if (qx * qx + qy * qy + qz * qz <= r2_expand3) {
                                if (compute_normals == 0) {
                                    mx += qx; my += qy; mz += qz;
                                    cxx += qx * qx; cxy += qx * qy; cxz += qx * qz;
                                    cyy += qy * qy; cyz += qy * qz; czz += qz * qz;
                                }
                                float njx = search_normals[3 * j + 0];
                                float njy = search_normals[3 * j + 1];
                                float njz = search_normals[3 * j + 2];
                                nxx += njx * njx; nxy += njx * njy; nxz += njx * njz;
                                nyy += njy * njy; nyz += njy * njz; nzz += njz * njz;
                                cnt++;
                            }
                        }
                    }
                }
            }
        }

        if (cnt < 4) {
            if (compute_normals == 0) {
                feat_out[6 * i + 0] = 0.11f;
                feat_out[6 * i + 1] = 0.30f;
                feat_out[6 * i + 2] = 0.22f;
                feat_out[6 * i + 3] = 0.20f;
                feat_out[6 * i + 4] = 0.18f;
            }
            feat_out[6 * i + 5] = 0.32f;
            continue;
        }

        double inv_n = 1.0 / cnt;
        if (compute_normals == 0) {
            mx *= inv_n; my *= inv_n; mz *= inv_n;
            Eigen::Matrix3f cov;
            cov(0, 0) = static_cast<float>(cxx * inv_n - mx * mx);
            cov(0, 1) = cov(1, 0) = static_cast<float>(cxy * inv_n - mx * my);
            cov(0, 2) = cov(2, 0) = static_cast<float>(cxz * inv_n - mx * mz);
            cov(1, 1) = static_cast<float>(cyy * inv_n - my * my);
            cov(1, 2) = cov(2, 1) = static_cast<float>(cyz * inv_n - my * mz);
            cov(2, 2) = static_cast<float>(czz * inv_n - mz * mz);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
            es.computeDirect(cov, Eigen::EigenvaluesOnly);
            const float l3 = std::max(0.0f, es.eigenvalues()(0));
            const float l2 = std::max(l3, es.eigenvalues()(1));
            const float l1 = std::max(l2, es.eigenvalues()(2));
            const float sum_l = l1 + l2 + l3 + 1e-12f;
            const float l1_eps = l1 + 1e-12f;

            feat_out[6 * i + 0] = l3 / sum_l;
            feat_out[6 * i + 1] = (l2 - l3) / l1_eps;
            feat_out[6 * i + 2] = l3 / l1_eps;
            feat_out[6 * i + 3] = (l1 - l2) / l1_eps;
            feat_out[6 * i + 4] = std::cbrt(l1 * l2 * l3) / sum_l;
        }

        // Sign-invariant normal structure tensor largest eigenvalue mu_max in [1/3, 1]
        Eigen::Matrix3f n_cov;
        n_cov(0, 0) = static_cast<float>(nxx * inv_n);
        n_cov(0, 1) = n_cov(1, 0) = static_cast<float>(nxy * inv_n);
        n_cov(0, 2) = n_cov(2, 0) = static_cast<float>(nxz * inv_n);
        n_cov(1, 1) = static_cast<float>(nyy * inv_n);
        n_cov(1, 2) = n_cov(2, 1) = static_cast<float>(nyz * inv_n);
        n_cov(2, 2) = static_cast<float>(nzz * inv_n);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> nes;
        nes.computeDirect(n_cov, Eigen::EigenvaluesOnly);
        const float mu_max = std::clamp(nes.eigenvalues()(2), 0.333333f, 1.0f);
        const float normal_dispersion = std::clamp(1.5f * (1.0f - mu_max), 0.0f, 1.0f);

        feat_out[6 * i + 5] = normal_dispersion;
    }
}

// -----------------------------------------------------------------------------
// 3. 3D Cone Enclosure & Cavern Roof/Floor Raycasting + Physical Normal Verification
// Computes 4 enclosure descriptors per point and enforces physical Floor/Ceiling
// normal consistency (Floor: nz > 0 when enclosed from above; Ceiling: nz < 0 when
// enclosed from below):
//   enclosure_out[i * 4 + 0] = upward_occlusion (fraction of +Z cone rays hitting any upper surface)
//   enclosure_out[i * 4 + 1] = upward_roof_occlusion (fraction hitting coherent rock roof/wall)
//   enclosure_out[i * 4 + 2] = downward_floor_occlusion (fraction of -Z cone rays hitting lower floor)
//   enclosure_out[i * 4 + 3] = cavern_enclosure_score (combined 3D tunnel/shaft enclosure in [0, 1])
// -----------------------------------------------------------------------------
void spt_compute_enclosure(
    const float* xyz,
    float* normals_inout,
    const float* fine_features,   // stride 6: [rug, plan, scat, lin, omni, ndisp] at r_fine
    const float* coarse_features, // stride 6: [rug, plan, scat, lin, omni, ndisp] at r_coarse
    int32_t n_pts,
    float occ_cell_size,
    float max_range,
    float* enclosure_out
) {
    SpatialGrid grid;
    grid.build(xyz, n_pts, occ_cell_size);
    const int32_t num_cells = static_cast<int32_t>(grid.point_count.size());

    // Precompute per-cell 2D rock surface weight using coarse-scale manifold coherence so
    // porous 3D leaf canopies have cell_rock_weight == 0.0 while coherent cave rock has ~ 1.0.
    // Also separate cell_roof_weight (blocks upward-facing hillside slopes nz > +0.15 from counting
    // as overhead cave ceilings) and cell_floor_weight (requires upward/horizontal floor surface).
    std::vector<float> cell_rock_weight(num_cells, 0.0f);
    std::vector<float> cell_roof_weight(num_cells, 0.0f);
    std::vector<float> cell_floor_weight(num_cells, 0.0f);

    #pragma omp parallel for schedule(static)
    for (int32_t c = 0; c < num_cells; ++c) {
        const int32_t start = grid.point_start[c];
        const int32_t count = grid.point_count[c];
        double c_rug = 0.0, c_plan = 0.0, c_scat = 0.0, c_disp = 0.0, c_nz = 0.0;
        for (int32_t k = 0; k < count; ++k) {
            int32_t idx = grid.sorted_points[start + k];
            c_rug  += coarse_features[6 * idx + 0];
            c_plan += coarse_features[6 * idx + 1];
            c_scat += coarse_features[6 * idx + 2];
            c_disp += coarse_features[6 * idx + 5];
            c_nz   += normals_inout[3 * idx + 2];
        }
        c_rug  /= count; c_plan /= count; c_scat /= count; c_disp /= count; c_nz /= count;

        // At coarse scale (r_coarse >= 0.50m), tree canopies never form a coherent 1m planar rock sheet,
        // so disp_w * scat_w * rug_w * plan_w == 0.0 in foliage canopies and ~1.0 on cave roofs/walls.
        float disp_w = std::clamp(1.0f - static_cast<float>(c_disp - 0.35) * 4.0f, 0.0f, 1.0f);
        float scat_w = std::clamp(1.0f - static_cast<float>(c_scat - 0.25) * 4.0f, 0.0f, 1.0f);
        float rug_w  = std::clamp(1.0f - static_cast<float>(c_rug  - 0.12) * 8.0f, 0.0f, 1.0f);
        float plan_w = std::clamp(static_cast<float>(c_plan - 0.32) * 3.0f, 0.0f, 1.0f);
        float rock_w = std::clamp(disp_w * scat_w * rug_w * plan_w, 0.0f, 1.0f);
        cell_rock_weight[c] = rock_w;
        cell_roof_weight[c]  = rock_w;
        cell_floor_weight[c] = rock_w;
    }

    // Generate 32 Fibonacci directions inside a 50-degree cone around +Z
    constexpr int K_CONE = 32;
    Eigen::Vector3f up_dirs[K_CONE];
    Eigen::Vector3f down_dirs[K_CONE];
    const float cos_max = std::cos(50.0f * static_cast<float>(M_PI) / 180.0f);
    const float golden_angle = static_cast<float>(M_PI * (3.0 - std::sqrt(5.0)));
    for (int k = 0; k < K_CONE; ++k) {
        float t = (k + 0.5f) / K_CONE;
        float cos_theta = 1.0f - t * (1.0f - cos_max);
        float sin_theta = std::sqrt(std::max(0.0f, 1.0f - cos_theta * cos_theta));
        float phi = k * golden_angle;
        up_dirs[k] = Eigen::Vector3f(sin_theta * std::cos(phi), sin_theta * std::sin(phi), cos_theta);
        down_dirs[k] = Eigen::Vector3f(sin_theta * std::cos(phi), sin_theta * std::sin(phi), -cos_theta);
    }

    // Generate 16 horizontal azimuth directions in XY plane for lateral wall/shaft enclosure
    constexpr int K_LAT = 16;
    Eigen::Vector3f lat_dirs[K_LAT];
    for (int k = 0; k < K_LAT; ++k) {
        float phi = 2.0f * static_cast<float>(M_PI) * k / K_LAT;
        lat_dirs[k] = Eigen::Vector3f(std::cos(phi), std::sin(phi), 0.0f);
    }

    const int max_steps = std::max(4, static_cast<int>(std::ceil(max_range / occ_cell_size)));
    const int min_step = std::max(4, static_cast<int>(std::ceil(1.00f / occ_cell_size)));

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int32_t i = 0; i < n_pts; ++i) {
        const float px = xyz[3 * i + 0];
        const float py = xyz[3 * i + 1];
        const float pz = xyz[3 * i + 2];
        const float self_c_rug  = coarse_features[6 * i + 0];
        const float self_c_plan = coarse_features[6 * i + 1];
        const float self_c_scat = coarse_features[6 * i + 2];
        const float self_c_disp = coarse_features[6 * i + 5];

        const float self_rock_coherence =
            std::clamp((self_c_plan - 0.32f) * 3.0f, 0.0f, 1.0f) *
            std::clamp(1.0f - (self_c_disp - 0.35f) * 4.0f, 0.0f, 1.0f) *
            std::clamp(1.0f - (self_c_scat - 0.25f) * 4.0f, 0.0f, 1.0f) *
            std::clamp(1.0f - (self_c_rug  - 0.12f) * 8.0f, 0.0f, 1.0f);

        int up_hits = 0;
        float roof_weighted_hits = 0.0f;

        for (int k = 0; k < K_CONE; ++k) {
            const Eigen::Vector3f& d = up_dirs[k];
            for (int s = min_step; s <= max_steps; ++s) {
                float dist = s * occ_cell_size;
                int32_t ix, iy, iz;
                grid.point_to_cell(px + d.x() * dist, py + d.y() * dist, pz + d.z() * dist, ix, iy, iz);
                int32_t cid = grid.find_cell(ix, iy, iz);
                if (cid != -1 && grid.point_count[cid] >= 1) {
                    up_hits++;
                    roof_weighted_hits += cell_roof_weight[cid];
                    break;
                }
            }
        }

        int down_hits = 0;
        float down_rock_hits = 0.0f;
        for (int k = 0; k < K_CONE; ++k) {
            const Eigen::Vector3f& d = down_dirs[k];
            for (int s = min_step; s <= max_steps; ++s) {
                float dist = s * occ_cell_size;
                int32_t ix, iy, iz;
                grid.point_to_cell(px + d.x() * dist, py + d.y() * dist, pz + d.z() * dist, ix, iy, iz);
                int32_t cid = grid.find_cell(ix, iy, iz);
                if (cid != -1 && grid.point_count[cid] >= 1) {
                    down_hits++;
                    down_rock_hits += cell_floor_weight[cid];
                    break;
                }
            }
        }

        int lat_rock_hits = 0;
        const int max_lat_steps = std::min(max_steps, static_cast<int>(std::ceil(25.0f / occ_cell_size)));
        for (int k = 0; k < K_LAT; ++k) {
            const Eigen::Vector3f& d = lat_dirs[k];
            for (int s = min_step; s <= max_lat_steps; ++s) {
                float dist = s * occ_cell_size;
                int32_t ix, iy, iz;
                grid.point_to_cell(px + d.x() * dist, py + d.y() * dist, pz, ix, iy, iz);
                int32_t cid = grid.find_cell(ix, iy, iz);
                if (cid != -1 && grid.point_count[cid] >= 1) {
                    if (cell_rock_weight[cid] > 0.45f) {
                        lat_rock_hits++;
                    }
                    break;
                }
            }
        }

        const float up_occ = static_cast<float>(up_hits) / K_CONE;
        const float roof_occ = roof_weighted_hits / K_CONE;
        const float down_occ = static_cast<float>(down_hits) / K_CONE;
        const float down_rock_occ = down_rock_hits / K_CONE;
        const float lat_occ = static_cast<float>(lat_rock_hits) / K_LAT;
        const float foliage_above = std::max(0.0f, up_occ - roof_occ);

        // Enforce physical vertical raycast consistency on Floor vs. Ceiling normals using ROCK occlusion:
        // - Floor has coherent rock ceiling above it (roof_occ > down_rock_occ + 0.12) -> normal must point UP (nz > 0)
        // - Ceiling has coherent rock floor below it (down_rock_occ > roof_occ + 0.15) AND is NOT under a foliage
        //   canopy (foliage_above < 0.30) -> normal must point DOWN (nz < 0)
        // - Exterior ground/slope under open sky or under foliage canopy -> normal must point UP (nz > 0)
        float nx = normals_inout[3 * i + 0];
        float ny = normals_inout[3 * i + 1];
        float nz = normals_inout[3 * i + 2];
        if (std::abs(nz) > 0.20f) {
            if (roof_occ > down_rock_occ + 0.12f && roof_occ > 0.12f && nz < 0.0f) {
                nx = -nx; ny = -ny; nz = -nz;
            } else if (down_rock_occ > roof_occ + 0.15f && down_rock_occ > 0.15f &&
                       (roof_occ > 0.10f || foliage_above < 0.55f) && self_rock_coherence > 0.30f && nz > 0.0f) {
                nx = -nx; ny = -ny; nz = -nz;
            } else if (roof_occ < 0.06f && down_rock_occ < 0.08f &&
                       (down_occ < 0.08f || (foliage_above > 0.60f && self_rock_coherence > 0.20f)) &&
                       nz < 0.0f) {
                nx = -nx; ny = -ny; nz = -nz;
            }
            normals_inout[3 * i + 0] = nx;
            normals_inout[3 * i + 1] = ny;
            normals_inout[3 * i + 2] = nz;
        }

        // Combined 3D cavern enclosure score:
        // 1. Floor / lower walls / internal speleothems: covered by a coherent rock ceiling (roof_occ)
        // 2. Overhanging ceiling vaults (nz < -0.10): 2D rock surface with floor below or lateral walls,
        //    damped to 0 if sitting beneath a dense 3D foliage canopy (foliage_above > 0.55 with roof_occ < 0.08)
        // 3. Vertical shafts / fissures: 2D rock walls enclosed laterally (lat_occ >= 0.50) and vertically
        float ceiling_vault_score = 0.0f;
        if (nz < -0.10f) {
            float overhang_strength = std::clamp((-nz - 0.10f) / 0.40f, 0.0f, 1.0f);
            float no_canopy_above = (roof_occ > 0.10f)
                ? 1.0f
                : std::clamp(1.0f - (foliage_above - 0.45f) * 3.0f, 0.0f, 1.0f);
            ceiling_vault_score = self_rock_coherence * overhang_strength * no_canopy_above *
                                  std::max(down_rock_occ, 0.7f * lat_occ);
        }
        float shaft_wall_score = self_rock_coherence * (
            std::sqrt(roof_occ * down_rock_occ) * 1.5f +
            0.70f * lat_occ * std::max({roof_occ, (lat_occ >= 0.50f ? down_rock_occ : 0.0f)})
        );

        float cavern_enclosure = std::clamp(
            std::max({roof_occ, ceiling_vault_score, shaft_wall_score}), 0.0f, 1.0f
        );

        enclosure_out[4 * i + 0] = up_occ;
        enclosure_out[4 * i + 1] = roof_occ;
        enclosure_out[4 * i + 2] = down_rock_occ;
        enclosure_out[4 * i + 3] = cavern_enclosure;
    }
}

// -----------------------------------------------------------------------------
// 4. Hierarchical l0-Cut Pursuit Superpoint Partitioning (P0 -> P1 -> P2)
// Partitions n_pts points with d_feat features into homogeneous superpoints by
// minimizing the l0-cut pursuit energy + minimum component size (cutoff).
// Implements:
//   1. Max-edge-length pruning (max_edge_dist = 3.5 * d_0) to sever air bridges
//      between overhanging branches and cave rock (keeping 1-NN only if isolated).
//   2. Contrast-sensitive edge weights w_ij combining spatial inverse distance,
//      normal alignment exp(-(1 - |n_i . n_j|) / sigma_n^2), and normal dispersion
//      jump exp(-(D_i - D_j)^2 / sigma_D^2) so superpoints snap cleanly to the
//      rock-foliage interface.
// -----------------------------------------------------------------------------
int32_t spt_partition_superpoints(
    const float* xyz,
    const float* normals,           // [n_pts, 3]
    const float* normal_dispersion, // [n_pts]
    const float* features,          // [n_pts, d_feat]
    int32_t n_pts,
    int32_t d_feat,
    float neighbor_radius,
    float max_edge_dist,
    float reg_strength,
    float spatial_weight,
    int32_t min_cutoff,
    int32_t* sp_labels_out
) {
    if (n_pts <= 0) return 0;

    SpatialGrid grid;
    grid.build(xyz, n_pts, neighbor_radius);
    const float r2 = neighbor_radius * neighbor_radius;
    const float max_edge_d2 = max_edge_dist * max_edge_dist;

    // Build pruned k=10 adjacency list + contrast-sensitive edge weights
    constexpr int MAX_K = 10;
    std::vector<int32_t> adj_indices(static_cast<size_t>(n_pts) * MAX_K, -1);
    std::vector<float> adj_weights(static_cast<size_t>(n_pts) * MAX_K, 1.0f);
    std::vector<int32_t> adj_counts(n_pts, 0);

    constexpr float inv_sigma_n2 = 1.0f / (0.35f * 0.35f);
    constexpr float inv_sigma_d2 = 1.0f / (0.20f * 0.20f);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (int32_t i = 0; i < n_pts; ++i) {
        const float px = xyz[3 * i + 0];
        const float py = xyz[3 * i + 1];
        const float pz = xyz[3 * i + 2];
        const float nix = normals[3 * i + 0];
        const float niy = normals[3 * i + 1];
        const float niz = normals[3 * i + 2];
        const float di = normal_dispersion[i];

        int32_t ix, iy, iz;
        grid.point_to_cell(px, py, pz, ix, iy, iz);

        std::pair<float, int32_t> cand[64];
        int n_cand = 0;

        for (int32_t dz = -1; dz <= 1; ++dz) {
            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                    if (cid == -1) continue;
                    const int32_t start = grid.point_start[cid];
                    const int32_t end = start + grid.point_count[cid];
                    for (int32_t k = start; k < end; ++k) {
                        int32_t j = grid.sorted_points[k];
                        if (j == i) continue;
                        float qx = xyz[3 * j + 0] - px;
                        float qy = xyz[3 * j + 1] - py;
                        float qz = xyz[3 * j + 2] - pz;
                        float d2 = qx * qx + qy * qy + qz * qz;
                        if (d2 <= r2) {
                            if (n_cand < 64) {
                                cand[n_cand++] = {d2, j};
                            } else if (d2 < cand[n_cand - 1].first) {
                                cand[n_cand - 1] = {d2, j};
                            }
                        }
                    }
                }
            }
        }
        std::sort(cand, cand + n_cand);

        // Prune edges longer than max_edge_dist (3.5 * d_0); keep single nearest neighbor if isolated
        int keep = 0;
        const int limit = std::min(n_cand, MAX_K);
        for (int k = 0; k < limit; ++k) {
            if (cand[k].first <= max_edge_d2 || k == 0) {
                int32_t j = cand[k].second;
                float dist = std::sqrt(cand[k].first);
                float njx = normals[3 * j + 0];
                float njy = normals[3 * j + 1];
                float njz = normals[3 * j + 2];
                float dj = normal_dispersion[j];

                float abs_dot = std::clamp(std::abs(nix * njx + niy * njy + niz * njz), 0.0f, 1.0f);
                float d_diff = di - dj;
                float w_norm = std::exp(-(1.0f - abs_dot) * inv_sigma_n2);
                float w_disp = std::exp(-(d_diff * d_diff) * inv_sigma_d2);
                float w_dist = 1.0f / (dist / std::max(max_edge_dist, 1e-3f) + 0.25f);
                float w_ij = std::clamp(w_dist * w_norm * w_disp, 0.03f, 4.0f);

                adj_indices[static_cast<size_t>(i) * MAX_K + keep] = j;
                adj_weights[static_cast<size_t>(i) * MAX_K + keep] = w_ij;
                keep++;
            } else {
                break;
            }
        }
        adj_counts[i] = keep;
    }

    // Disjoint-set (Union-Find) with component feature centroids & sizes for l0-cut pursuit
    std::vector<int32_t> parent(n_pts);
    std::vector<int32_t> comp_size(n_pts, 1);
    std::vector<float> comp_feat(static_cast<size_t>(n_pts) * d_feat);
    std::vector<float> comp_xyz(static_cast<size_t>(n_pts) * 3);

    for (int32_t i = 0; i < n_pts; ++i) {
        parent[i] = i;
        for (int d = 0; d < d_feat; ++d) {
            comp_feat[static_cast<size_t>(i) * d_feat + d] = features[static_cast<size_t>(i) * d_feat + d];
        }
        comp_xyz[3 * i + 0] = xyz[3 * i + 0];
        comp_xyz[3 * i + 1] = xyz[3 * i + 1];
        comp_xyz[3 * i + 2] = xyz[3 * i + 2];
    }

    auto find_root = [&](int32_t u) {
        while (parent[u] != u) {
            parent[u] = parent[parent[u]];
            u = parent[u];
        }
        return u;
    };

    auto merge_Select = [&](int32_t ru, int32_t rv) {
        if (ru == rv) return ru;
        if (comp_size[ru] < comp_size[rv]) std::swap(ru, rv);
        const float su = static_cast<float>(comp_size[ru]);
        const float sv = static_cast<float>(comp_size[rv]);
        const float inv_s = 1.0f / (su + sv);
        for (int d = 0; d < d_feat; ++d) {
            comp_feat[static_cast<size_t>(ru) * d_feat + d] =
                (comp_feat[static_cast<size_t>(ru) * d_feat + d] * su +
                 comp_feat[static_cast<size_t>(rv) * d_feat + d] * sv) * inv_s;
        }
        for (int d = 0; d < 3; ++d) {
            comp_xyz[3 * ru + d] = (comp_xyz[3 * ru + d] * su + comp_xyz[3 * rv + d] * sv) * inv_s;
        }
        parent[rv] = ru;
        comp_size[ru] += comp_size[rv];
        return ru;
    };

    // Multi-pass l0-cut pursuit region merging with progressively increasing threshold up to reg_strength
    const float thresholds[4] = {
        reg_strength * 0.15f,
        reg_strength * 0.40f,
        reg_strength * 0.75f,
        reg_strength
    };

    for (int pass = 0; pass < 4; ++pass) {
        const float thresh = thresholds[pass];
        for (int32_t i = 0; i < n_pts; ++i) {
            int32_t ru = find_root(i);
            const int deg = adj_counts[i];
            for (int k = 0; k < deg; ++k) {
                int32_t j = adj_indices[static_cast<size_t>(i) * MAX_K + k];
                float w_ij = adj_weights[static_cast<size_t>(i) * MAX_K + k];
                int32_t rv = find_root(j);
                if (ru == rv) continue;

                // Compute contrast-weighted l0-cut pursuit merge cost between superpoints ru and rv:
                // Low edge affinity w_ij across normal/dispersion jumps inflates effective cost, preventing merge
                float feat_dist2 = 0.0f;
                for (int d = 0; d < d_feat; ++d) {
                    float diff = comp_feat[static_cast<size_t>(ru) * d_feat + d] -
                                 comp_feat[static_cast<size_t>(rv) * d_feat + d];
                    feat_dist2 += diff * diff;
                }
                float dx = comp_xyz[3 * ru + 0] - comp_xyz[3 * rv + 0];
                float dy = comp_xyz[3 * ru + 1] - comp_xyz[3 * rv + 1];
                float dz = comp_xyz[3 * ru + 2] - comp_xyz[3 * rv + 2];
                float sp_dist2 = (dx * dx + dy * dy + dz * dz) * spatial_weight;

                float total_cost = (feat_dist2 + sp_dist2) / std::max(w_ij, 0.05f);
                if (total_cost <= thresh) {
                    ru = merge_Select(ru, rv);
                }
            }
        }
    }

    // Enforce minimum component size (min_cutoff) by merging small fragments into most compatible neighbor
    for (int iter = 0; iter < 3; ++iter) {
        for (int32_t i = 0; i < n_pts; ++i) {
            int32_t ru = find_root(i);
            if (comp_size[ru] >= min_cutoff) continue;

            const int deg = adj_counts[i];
            int32_t best_rv = -1;
            float best_cost = 1e30f;
            for (int k = 0; k < deg; ++k) {
                int32_t j = adj_indices[static_cast<size_t>(i) * MAX_K + k];
                float w_ij = adj_weights[static_cast<size_t>(i) * MAX_K + k];
                int32_t rv = find_root(j);
                if (ru == rv) continue;

                float feat_dist2 = 0.0f;
                for (int d = 0; d < d_feat; ++d) {
                    float diff = comp_feat[static_cast<size_t>(ru) * d_feat + d] -
                                 comp_feat[static_cast<size_t>(rv) * d_feat + d];
                    feat_dist2 += diff * diff;
                }
                float weighted_cost = feat_dist2 / std::max(w_ij, 0.05f);
                if (weighted_cost < best_cost) {
                    best_cost = weighted_cost;
                    best_rv = rv;
                }
            }
            if (best_rv != -1) {
                merge_Select(ru, best_rv);
            }
        }
    }

    // Relabel components contiguously 0 .. num_sp - 1
    std::vector<int32_t> root_to_id(n_pts, -1);
    int32_t num_sp = 0;
    for (int32_t i = 0; i < n_pts; ++i) {
        int32_t r = find_root(i);
        if (root_to_id[r] == -1) {
            root_to_id[r] = num_sp++;
        }
        sp_labels_out[i] = root_to_id[r];
    }
    return num_sp;
}

// -----------------------------------------------------------------------------
// 5. Exact Superpoint Boundary Adjacency Graph Extraction
// Connects superpoints u and v if any constituent P0 points lie within adj_radius.
// Writes up to max_edges undirected edges (u < v) to edge_u_out and edge_v_out.
// -----------------------------------------------------------------------------
int32_t spt_build_superpoint_graph(
    const float* xyz,
    const int32_t* sp_labels,
    int32_t n_pts,
    int32_t num_sp,
    float adj_radius,
    int32_t max_edges,
    int32_t* edge_u_out,
    int32_t* edge_v_out
) {
    if (n_pts <= 0 || num_sp <= 1) return 0;

    SpatialGrid grid;
    grid.build(xyz, n_pts, adj_radius);
    const float r2 = adj_radius * adj_radius;

    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<uint64_t>> thread_edges(n_threads);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& local_vec = thread_edges[tid];
        local_vec.reserve(4096);

        #pragma omp for schedule(dynamic, 1024)
        for (int32_t i = 0; i < n_pts; ++i) {
            const int32_t sp_u = sp_labels[i];
            const float px = xyz[3 * i + 0];
            const float py = xyz[3 * i + 1];
            const float pz = xyz[3 * i + 2];
            int32_t ix, iy, iz;
            grid.point_to_cell(px, py, pz, ix, iy, iz);

            for (int32_t dz = -1; dz <= 1; ++dz) {
                for (int32_t dy = -1; dy <= 1; ++dy) {
                    for (int32_t dx = -1; dx <= 1; ++dx) {
                        int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                        if (cid == -1) continue;
                        const int32_t start = grid.point_start[cid];
                        const int32_t end = start + grid.point_count[cid];
                        for (int32_t k = start; k < end; ++k) {
                            int32_t j = grid.sorted_points[k];
                            int32_t sp_v = sp_labels[j];
                            if (sp_u >= sp_v) continue;
                            float qx = xyz[3 * j + 0] - px;
                            float qy = xyz[3 * j + 1] - py;
                            float qz = xyz[3 * j + 2] - pz;
                            if (qx * qx + qy * qy + qz * qz <= r2) {
                                uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(sp_u)) << 32ULL) |
                                               static_cast<uint64_t>(static_cast<uint32_t>(sp_v));
                                local_vec.push_back(key);
                            }
                        }
                    }
                }
            }
        }
        std::sort(local_vec.begin(), local_vec.end());
        local_vec.erase(std::unique(local_vec.begin(), local_vec.end()), local_vec.end());
    }

    std::vector<uint64_t> all_edges;
    for (int t = 0; t < n_threads; ++t) {
        all_edges.insert(all_edges.end(), thread_edges[t].begin(), thread_edges[t].end());
    }
    std::sort(all_edges.begin(), all_edges.end());
    all_edges.erase(std::unique(all_edges.begin(), all_edges.end()), all_edges.end());

    const int32_t count = std::min(static_cast<int32_t>(all_edges.size()), max_edges);
    for (int32_t e = 0; e < count; ++e) {
        edge_u_out[e] = static_cast<int32_t>(all_edges[e] >> 32ULL);
        edge_v_out[e] = static_cast<int32_t>(all_edges[e] & 0xFFFFFFFFULL);
    }
    return count;
}

// -----------------------------------------------------------------------------
// 6. Fast Out-of-Core Mapping of Full-Resolution Point Chunks to Nearest P0 Voxel
// Used during chunked streaming of 100M-300M+ point clouds to map P0 predictions
// and normals back to full-resolution chunks in O(N_chunk) time and bounded RAM.
// -----------------------------------------------------------------------------
void spt_map_points_to_p0(
    const float* p0_xyz,
    int32_t num_p0,
    const float* query_xyz,
    int32_t n_query,
    float voxel_size,
    int32_t* query_to_p0_out
) {
    if (num_p0 <= 0 || n_query <= 0) return;

    SpatialGrid grid;
    grid.build(p0_xyz, num_p0, voxel_size * 1.8f);

    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < n_query; ++i) {
        const float px = query_xyz[3 * i + 0];
        const float py = query_xyz[3 * i + 1];
        const float pz = query_xyz[3 * i + 2];
        int32_t ix, iy, iz;
        grid.point_to_cell(px, py, pz, ix, iy, iz);

        int32_t best_j = 0;
        float best_d2 = 1e30f;

        for (int32_t dz = -1; dz <= 1; ++dz) {
            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    int32_t cid = grid.find_cell(ix + dx, iy + dy, iz + dz);
                    if (cid == -1) continue;
                    const int32_t start = grid.point_start[cid];
                    const int32_t end = start + grid.point_count[cid];
                    for (int32_t k = start; k < end; ++k) {
                        int32_t j = grid.sorted_points[k];
                        float dx_ = p0_xyz[3 * j + 0] - px;
                        float dy_ = p0_xyz[3 * j + 1] - py;
                        float dz_ = p0_xyz[3 * j + 2] - pz;
                        float d2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_j = j;
                        }
                    }
                }
            }
        }
        query_to_p0_out[i] = best_j;
    }
}

// -----------------------------------------------------------------------------
// 7. CUDA-Accelerated WNNC Global Normal Estimation (Lin et al., ACM ToG 2024)
// Uses WindingNumberOperator<float> + wnnc::estimateNormals to compute globally
// consistent normals across the cave point cloud.
// Negates outward WNNC dipole normals so interior cave normals point INWARD:
//   - Cave Ceiling : nz < 0 (pointing downward into the cavity)
//   - Cave Floor   : nz > 0 (pointing upward into the cavity)
//   - Cave Walls   : |nz| ~ 0 (pointing horizontally into the cavity)
// For massive P0 clouds (n_pts > max_wnnc_pts), solves WNNC on a uniform carrier
// cloud of ~max_wnnc_pts points and propagates/interpolates to all P0 points.
// -----------------------------------------------------------------------------
void spt_compute_wnnc_normals(
    const float* xyz,
    int32_t n_pts,
    int32_t iterations,
    float width_min,
    float width_max,
    int32_t max_wnnc_pts,
    float* normals_out
) {
    if (n_pts <= 0) return;

    auto run_wnnc_direct = [&](const float* pts_in, int32_t count, float* nrm_out) {
        float min_x = pts_in[0], min_y = pts_in[1], min_z = pts_in[2];
        float max_x = min_x, max_y = min_y, max_z = min_z;
        for (int32_t i = 1; i < count; ++i) {
            min_x = std::min(min_x, pts_in[3 * i + 0]);
            min_y = std::min(min_y, pts_in[3 * i + 1]);
            min_z = std::min(min_z, pts_in[3 * i + 2]);
            max_x = std::max(max_x, pts_in[3 * i + 0]);
            max_y = std::max(max_y, pts_in[3 * i + 1]);
            max_z = std::max(max_z, pts_in[3 * i + 2]);
        }
        const float cx = 0.5f * (min_x + max_x);
        const float cy = 0.5f * (min_y + max_y);
        const float cz = 0.5f * (min_z + max_z);
        const float longest = std::max({max_x - min_x, max_y - min_y, max_z - min_z, 1e-6f});
        const float scale = 2.0f / (longest * 1.1f);

        std::vector<wnnc::Vec3<float>> norm_pts(count);
        #pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < count; ++i) {
            norm_pts[i] = {
                (pts_in[3 * i + 0] - cx) * scale,
                (pts_in[3 * i + 1] - cy) * scale,
                (pts_in[3 * i + 2] - cz) * scale
            };
        }

        wnnc::WindingNumberOperator<float> wn_op(std::move(norm_pts), 15, 4);
        wnnc::SolverSettings settings;
        settings.iterations = iterations;
        settings.widthMin = width_min;
        settings.widthMax = width_max;

        std::vector<wnnc::Vec3<float>> wn_normals = wnnc::estimateNormals(wn_op, settings);

        // Negate outward WNNC normal so cave normals point inward into the cavity
        // (Ceiling: nz < 0, Floor: nz > 0)
        #pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < count; ++i) {
            nrm_out[3 * i + 0] = -wn_normals[i].x;
            nrm_out[3 * i + 1] = -wn_normals[i].y;
            nrm_out[3 * i + 2] = -wn_normals[i].z;
        }
    };

    if (n_pts <= max_wnnc_pts) {
        run_wnnc_direct(xyz, n_pts, normals_out);
        return;
    }

    // Subsample a uniform carrier grid targeting ~max_wnnc_pts points
    float min_x = xyz[0], min_y = xyz[1], min_z = xyz[2];
    float max_x = min_x, max_y = min_y, max_z = min_z;
    for (int32_t i = 1; i < n_pts; ++i) {
        min_x = std::min(min_x, xyz[3 * i + 0]);
        min_y = std::min(min_y, xyz[3 * i + 1]);
        min_z = std::min(min_z, xyz[3 * i + 2]);
        max_x = std::max(max_x, xyz[3 * i + 0]);
        max_y = std::max(max_y, xyz[3 * i + 1]);
        max_z = std::max(max_z, xyz[3 * i + 2]);
    }
    const float diag = std::hypot(max_x - min_x, std::hypot(max_y - min_y, max_z - min_z));
    float carrier_voxel = std::max(0.08f, diag / 900.0f);

    SpatialGrid cgrid;
    cgrid.build(xyz, n_pts, carrier_voxel);
    while (static_cast<int32_t>(cgrid.point_count.size()) > max_wnnc_pts) {
        carrier_voxel *= 1.25f;
        cgrid.build(xyz, n_pts, carrier_voxel);
    }

    const int32_t n_carr = static_cast<int32_t>(cgrid.point_count.size());
    std::vector<float> carr_xyz(static_cast<size_t>(n_carr) * 3);
    std::vector<float> carr_nrm(static_cast<size_t>(n_carr) * 3);

    #pragma omp parallel for schedule(static)
    for (int32_t c = 0; c < n_carr; ++c) {
        const int32_t start = cgrid.point_start[c];
        const int32_t count = cgrid.point_count[c];
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (int32_t k = 0; k < count; ++k) {
            int32_t idx = cgrid.sorted_points[start + k];
            cx += xyz[3 * idx + 0];
            cy += xyz[3 * idx + 1];
            cz += xyz[3 * idx + 2];
        }
        carr_xyz[3 * c + 0] = static_cast<float>(cx / count);
        carr_xyz[3 * c + 1] = static_cast<float>(cy / count);
        carr_xyz[3 * c + 2] = static_cast<float>(cz / count);
    }

    run_wnnc_direct(carr_xyz.data(), n_carr, carr_nrm.data());

    // Smoothly interpolate WNNC carrier normals to all P0 points via inverse-distance weighting
    SpatialGrid ngrid;
    ngrid.build(carr_xyz.data(), n_carr, carrier_voxel * 1.6f);

    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < n_pts; ++i) {
        const float px = xyz[3 * i + 0];
        const float py = xyz[3 * i + 1];
        const float pz = xyz[3 * i + 2];
        int32_t ix, iy, iz;
        ngrid.point_to_cell(px, py, pz, ix, iy, iz);

        Eigen::Vector3f acc(0.0f, 0.0f, 0.0f);
        int32_t best_j = 0;
        float best_d2 = 1e30f;

        for (int32_t dz = -1; dz <= 1; ++dz) {
            for (int32_t dy = -1; dy <= 1; ++dy) {
                for (int32_t dx = -1; dx <= 1; ++dx) {
                    int32_t cid = ngrid.find_cell(ix + dx, iy + dy, iz + dz);
                    if (cid == -1) continue;
                    const int32_t start = ngrid.point_start[cid];
                    const int32_t end = start + ngrid.point_count[cid];
                    for (int32_t k = start; k < end; ++k) {
                        int32_t j = ngrid.sorted_points[k];
                        float dx_ = carr_xyz[3 * j + 0] - px;
                        float dy_ = carr_xyz[3 * j + 1] - py;
                        float dz_ = carr_xyz[3 * j + 2] - pz;
                        float d2 = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                        if (d2 < best_d2) {
                            best_d2 = d2;
                            best_j = j;
                        }
                        float w = 1.0f / (d2 + 0.0025f);
                        acc.x() += w * carr_nrm[3 * j + 0];
                        acc.y() += w * carr_nrm[3 * j + 1];
                        acc.z() += w * carr_nrm[3 * j + 2];
                    }
                }
            }
        }
        float norm = acc.norm();
        if (norm > 1e-6f) {
            acc /= norm;
            normals_out[3 * i + 0] = acc.x();
            normals_out[3 * i + 1] = acc.y();
            normals_out[3 * i + 2] = acc.z();
        } else {
            normals_out[3 * i + 0] = carr_nrm[3 * best_j + 0];
            normals_out[3 * i + 1] = carr_nrm[3 * best_j + 1];
            normals_out[3 * i + 2] = carr_nrm[3 * best_j + 2];
        }
    }
}

} // extern "C"


