// SPDX-License-Identifier: MIT
// Unified pure C++ pipeline main coordinator
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <fstream>
#include <future>
#include <thread>
#include <system_error>

#include <CLI11.hpp>

// CGAL headers from preprocess
#include "mesh_reconstruction.h"
#include "point_cloud_processing.h"
#include "skeleton_extraction.h"
#include "io_visualization.h"

// WNNC headers
#include "wnnc/normal_solver.h"
#include "wnnc/point_cloud_io.h"
#include "wnnc/vec3.h"
#include "wnnc/winding_number_operator.h"

// Poisson/Trim wrapper
#include "poisson_recon_wrapper.h"

#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#if __has_include(<CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_stop_predicate.h>)
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_count_stop_predicate.h>
using Stop_predicate = CGAL::Surface_mesh_simplification::Edge_count_stop_predicate<mesh_reconstruction::Triangle_mesh>;
#elif __has_include(<CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_stop_predicate.h>)
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_stop_predicate.h>
using Stop_predicate = CGAL::Surface_mesh_simplification::Count_stop_predicate<mesh_reconstruction::Triangle_mesh>;
#else
#error "CGAL stop predicate header not found"
#endif

namespace fs = std::filesystem;

namespace {

const std::map<std::string, std::pair<double, double>> kPresetWidths = {
    {"l0", {0.002, 0.016}}, {"l1", {0.01, 0.04}}, {"l2", {0.02, 0.08}},
    {"l3", {0.03, 0.12}},   {"l4", {0.04, 0.16}}, {"l5", {0.05, 0.2}},
};

template <typename Scalar>
std::vector<wnnc::Vec3<Scalar>> normalizeToUnitCube(const std::vector<wnnc::Vec3<double>>& points) {
    constexpr double kBoundingBoxScale = 1.1;

    wnnc::Vec3<double> boxMin = points.front();
    wnnc::Vec3<double> boxMax = points.front();
    for (const wnnc::Vec3<double>& p : points) {
        boxMin = {std::min(boxMin.x, p.x), std::min(boxMin.y, p.y), std::min(boxMin.z, p.z)};
        boxMax = {std::max(boxMax.x, p.x), std::max(boxMax.y, p.y), std::max(boxMax.z, p.z)};
    }
    const wnnc::Vec3<double> center = (boxMin + boxMax) * 0.5;
    const wnnc::Vec3<double> extent = boxMax - boxMin;
    const double longestSide = std::max({extent.x, extent.y, extent.z});
    const double scale = 2.0 / (longestSide * kBoundingBoxScale);

    std::vector<wnnc::Vec3<Scalar>> normalized;
    normalized.reserve(points.size());
    for (const wnnc::Vec3<double>& p : points) {
        normalized.push_back(((p - center) * scale).template cast<Scalar>());
    }
    return normalized;
}

// Basic PLY exporter for PipelineMesh
bool save_mesh_ply(const fs::path& filepath, const PipelineMesh& mesh) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }
    file << "ply\n";
    file << "format ascii 1.0\n";
    file << "element vertex " << mesh.vertices.size() << "\n";
    file << "property double x\n";
    file << "property double y\n";
    file << "property double z\n";
    file << "property double value\n";
    file << "element face " << mesh.faces.size() << "\n";
    file << "property list uchar int vertex_indices\n";
    file << "end_header\n";
    for (const auto& v : mesh.vertices) {
        file << v.x << " " << v.y << " " << v.z << " " << v.value << "\n";
    }
    for (const auto& f : mesh.faces) {
        file << f.size();
        for (auto idx : f) {
            file << " " << idx;
        }
        file << "\n";
    }
    return true;
}

template <typename Func>
void spawn_bg_task(std::vector<std::future<void>>& bg_tasks, Func&& func) {
    bg_tasks.push_back(std::async(std::launch::async, [f = std::forward<Func>(func)]() {
        try {
            f();
        } catch (const std::exception& e) {
            std::cerr << "\n[Background I/O Error] " << e.what() << "\n" << std::flush;
        } catch (...) {
            std::cerr << "\n[Background I/O Error] Unknown exception occurred.\n" << std::flush;
        }
    }));
}

} // namespace

int main(int argc, char* argv[]) {
    CLI::App app{"Unified C++ reconstruction and skeletonization pipeline (no Python)"};

    // Paths
    std::string input_path;
    std::string output_root = "pipeline_output";

    // Preprocessing options
    Pipeline_options prep_opts;
    // WNNC options
    wnnc::SolverSettings wnnc_opts;
    std::string widthConfig = "l0";
    double customWidthMin = 0.002;
    double customWidthMax = 0.016;
    bool wnnc_progress = false;

    // Poisson/Trimming options
    PoissonParams poisson_opts;
    TrimParams trim_opts;
    bool enable_trim = false;

    // Debugging / Save intermediate flags
    bool save_intermediate = false;
    bool save_stage1 = false;
    bool save_stage2 = false;
    bool save_stage3 = false;

    // CLI configuration
    app.add_option("input", input_path, "Input point cloud PLY file")->required()->check(CLI::ExistingFile);
    app.add_option("output_dir", output_root, "Output root directory")->capture_default_str();

    // Preprocessor CLI options
    auto prep_group = app.add_option_group("Preprocessor Options");
    prep_group->add_flag("--enable-spatial-subsampling", prep_opts.enable_spatial_subsampling, "Enable spatial subsampling");
    prep_group->add_option("--spatial-subsample-distance", prep_opts.spatial_subsample_distance, "Spatial subsampling distance")->capture_default_str();
    prep_group->add_option("--remove-outliers-percent", prep_opts.outlier_percent, "Percentage threshold for outlier removal")->capture_default_str();
    prep_group->add_option("--outlier-neighbors", prep_opts.outlier_neighbors, "Number of neighbors for outlier checks")->capture_default_str();
    prep_group->add_flag("--enable-wlop", prep_opts.enable_wlop, "Enable WLOP downsampling");
    prep_group->add_option("--wlop-retain-percent", prep_opts.wlop_retain_percent, "Percentage of points to retain in WLOP")->capture_default_str();
    prep_group->add_option("--wlop-neighbor-radius", prep_opts.wlop_neighbor_radius, "Neighbor radius for WLOP")->capture_default_str();
    prep_group->add_option("--wlop-iterations", prep_opts.wlop_iterations, "Number of WLOP iterations")->capture_default_str();
    prep_group->add_flag("--wlop-require-uniform-sampling", prep_opts.wlop_require_uniform_sampling, "Require uniform sampling in WLOP");
    prep_group->add_flag("--enable-smoothing", prep_opts.enable_smoothing, "Enable jet smoothing");
    prep_group->add_option("--smoothing-neighbors", prep_opts.smoothing_neighbors, "Number of neighbors for jet smoothing")->capture_default_str();

    // WNNC CLI options
    auto wnnc_group = app.add_option_group("WNNC Normal Estimation Options");
    wnnc_group->add_option("--width_config", widthConfig, "Smoothing width preset (l0...l5) or 'custom'")
        ->capture_default_str()->check(CLI::IsMember({"l0", "l1", "l2", "l3", "l4", "l5", "custom"}));
    wnnc_group->add_option("--wsmin", customWidthMin, "Custom minimum smoothing width (requires 'custom' width_config)")->capture_default_str();
    wnnc_group->add_option("--wsmax", customWidthMax, "Custom maximum smoothing width (requires 'custom' width_config)")->capture_default_str();
    wnnc_group->add_option("--wnnc-iters", wnnc_opts.iterations, "Number of WNNC solver iterations")->capture_default_str();
    wnnc_group->add_flag("--wnnc-progress", wnnc_progress, "Show WNNC iteration progress");

    // Poisson CLI options
    auto poisson_group = app.add_option_group("Poisson Reconstruction Options");
    poisson_group->add_option("--depth", poisson_opts.depth, "Poisson tree reconstruction depth")->capture_default_str();
    poisson_group->add_option("--solveDepth", poisson_opts.solveDepth, "Poisson solve depth")->capture_default_str();
    poisson_group->add_option("--samplesPerNode", poisson_opts.samplesPerNode, "Minimum number of samples per node")->capture_default_str();
    poisson_group->add_option("--pointWeight", poisson_opts.pointWeight, "Screened Poisson interpolation weight")->capture_default_str();
    poisson_group->add_option("--scale", poisson_opts.scale, "Scale factor")->capture_default_str();
    poisson_group->add_option("--bType", poisson_opts.bType, "Boundary type (1=free, 2=dirichlet, 3=neumann)")->capture_default_str();
    poisson_group->add_option("--degree", poisson_opts.degree, "B-spline degree")->capture_default_str();

    // Trimming CLI options
    auto trim_group = app.add_option_group("Trimming Options");
    trim_group->add_flag("--enable-trim", enable_trim, "Enable Stage 4 surface trimming");
    trim_group->add_option("--trim-threshold", trim_opts.trimThreshold, "Trimming threshold density value")->capture_default_str();
    trim_group->add_option("--aRatio", trim_opts.islandAreaRatio, "Relative area of islands to retain")->capture_default_str();
    trim_group->add_flag("--removeIslands", trim_opts.removeIslands, "Remove small disconnected components")->capture_default_str();

    // Debugging / Intermediates
    auto debug_group = app.add_option_group("Debugging Options");
    debug_group->add_flag("--save-intermediate", save_intermediate, "Save all intermediate stage outputs");
    debug_group->add_flag("--save-stage1", save_stage1, "Save preprocessed points PLY (Stage 1)");
    debug_group->add_flag("--save-stage2", save_stage2, "Save point-normals XYZ (Stage 2)");
    debug_group->add_flag("--save-stage3", save_stage3, "Save untrimmed Poisson mesh PLY (Stage 3)");

    CLI11_PARSE(app, argc, argv);

    // Apply width config logic for WNNC
    if (widthConfig == "custom") {
        wnnc_opts.widthMin = customWidthMin;
        wnnc_opts.widthMax = customWidthMax;
    } else {
        std::tie(wnnc_opts.widthMin, wnnc_opts.widthMax) = kPresetWidths.at(widthConfig);
    }
    if (wnnc_opts.widthMin > wnnc_opts.widthMax) {
        std::cerr << "Error: wsmin must not exceed wsmax\n";
        return EXIT_FAILURE;
    }

    if (save_intermediate) {
        save_stage1 = true;
        save_stage2 = true;
        save_stage3 = true;
    }

    prep_opts.input_path = input_path;
    prep_opts.output_dir = output_root;

    // Create directory structure mirroring the original Python stage directories
    fs::path root_path(output_root);
    fs::path stage1_dir = root_path / "stage1_preprocess";
    fs::path stage2_dir = root_path / "stage2_wnnc";
    fs::path stage3_dir = root_path / "stage3_surface";
    fs::path stage4_dir = root_path / "stage4_trimmed";
    fs::path stage5_dir = root_path / "stage5_skeleton";

    for (const auto& dir : {stage1_dir, stage2_dir, stage3_dir, stage4_dir, stage5_dir}) {
        fs::create_directories(dir);
    }

    fs::path input_fs_path(input_path);
    std::string stem = input_fs_path.stem().string();

    std::cout << "Starting Unified C++ Reconstruction Pipeline for: " << input_path << "\n";

    // Vector to track background asynchronous tasks
    std::vector<std::future<void>> bg_tasks;

    auto wait_bg_tasks = [&bg_tasks]() {
        for (auto& task : bg_tasks) {
            if (task.valid()) {
                task.wait();
            }
        }
    };

    // -------------------------------------------------------------------------
    // STAGE 1: Preprocessing using CGAL
    // -------------------------------------------------------------------------
    auto start_time = std::chrono::steady_clock::now();
    std::cout << "\n--- [Stage 1] Running Point Cloud Preprocessing ---\n";
    mesh_reconstruction::Point_set points;
    if (!load_oriented_points(input_path, points, prep_opts)) {
        std::cerr << "Error during Stage 1 point cloud loading.\n";
        wait_bg_tasks();
        return EXIT_FAILURE;
    }

    double average_spacing = 0.0;
    if (!preprocess_points(points, prep_opts, average_spacing)) {
        std::cerr << "Error during Stage 1 point cloud preprocessing.\n";
        wait_bg_tasks();
        return EXIT_FAILURE;
    }

    if (save_stage1) {
        fs::path s1_out_dir = stage1_dir / stem;
        fs::create_directories(s1_out_dir);
        fs::path s1_out_file = s1_out_dir / (stem + "_stage1_preprocessed_points.ply");
        spawn_bg_task(bg_tasks, [s1_out_file, points_copy = points]() {
            std::cout << "Saving Stage 1 output to: " << s1_out_file << "\n";
            if (!write_point_stage_visualization(s1_out_file, points_copy, "Preprocessed Point Cloud")) {
                std::cerr << "Warning: Failed to save Stage 1 preprocessed points.\n";
            }
        });
    }

    // -------------------------------------------------------------------------
    // STAGE 2: Normal Estimation using WNNC (C++)
    // -------------------------------------------------------------------------
    std::cout << "\n--- [Stage 2] Running WNNC Normal Estimation ---\n";
    std::vector<wnnc::Vec3<double>> rawPoints;
    rawPoints.reserve(points.size());
    for (auto it = points.begin(); it != points.end(); ++it) {
        auto p = points.point(*it);
        rawPoints.push_back({p.x(), p.y(), p.z()});
    }

    std::cout << "Running WNNC with " << rawPoints.size() << " points.\n";
    auto wnnc_start = std::chrono::steady_clock::now();
    
    // Normalize coordinates to unit cube for the octree and kernel presets
    std::vector<wnnc::Vec3<double>> normalized = normalizeToUnitCube<double>(rawPoints);
    wnnc::WindingNumberOperator<double> windingNumber(normalized);

    auto onIteration = [wnnc_progress](int done, int total) {
        if (!wnnc_progress) return;
        std::cerr << "\r[WNNC] iteration " << done << "/" << total << std::flush;
        if (done == total) std::cerr << '\n';
    };

    std::vector<wnnc::Vec3<double>> normals = wnnc::estimateNormals(windingNumber, wnnc_opts, onIteration);
    
    double wnnc_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - wnnc_start).count();
    std::cout << "WNNC finished. Duration: " << wnnc_duration << " seconds.\n";

    // -------------------------------------------------------------------------
    // STAGE 3: Screened Poisson Reconstruction (In-Memory)
    // -------------------------------------------------------------------------
    std::cout << "\n--- [Stage 3] Running Screened Poisson Reconstruction ---\n";
    std::vector<PipelinePoint> pipeline_pts(rawPoints.size());
    for (size_t i = 0; i < rawPoints.size(); ++i) {
        pipeline_pts[i] = {rawPoints[i].x, rawPoints[i].y, rawPoints[i].z, normals[i].x, normals[i].y, normals[i].z};
    }

    // Now that pipeline_pts are constructed, we can move rawPoints and normals to the background thread
    if (save_stage2) {
        fs::path s2_out_dir = stage2_dir / stem;
        fs::create_directories(s2_out_dir);
        fs::path s2_out_file = s2_out_dir / (stem + ".xyz");
        spawn_bg_task(bg_tasks, [s2_out_file, pts = std::move(rawPoints), nls = std::move(normals)]() {
            std::cout << "Saving WNNC outputs to: " << s2_out_file << "\n";
            wnnc::io::saveXyzWithNormals(s2_out_file, pts, nls);
        });
    }

    auto poisson_start = std::chrono::steady_clock::now();
    PipelineMesh untrimmed_mesh = run_poisson_reconstruction(pipeline_pts, poisson_opts);
    double poisson_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - poisson_start).count();
    std::cout << "Poisson reconstruction finished in memory. Duration: " << poisson_duration << " seconds.\n";
    std::cout << "Untrimmed watertight mesh has " << untrimmed_mesh.vertices.size() << " vertices, " << untrimmed_mesh.faces.size() << " faces.\n";

    fs::path s3_out_file = stage3_dir / (stem + "_watertight.ply");
    if (save_stage3) {
        spawn_bg_task(bg_tasks, [s3_out_file, mesh = untrimmed_mesh]() {
            std::cout << "Saving untrimmed watertight mesh to: " << s3_out_file << "\n";
            if (!save_mesh_ply(s3_out_file, mesh)) {
                std::cerr << "Warning: Failed to save untrimmed watertight mesh.\n";
            }
        });
    }

    // -------------------------------------------------------------------------
    // STAGE 4: Surface Trimming (Optional, In-Memory)
    // -------------------------------------------------------------------------
    PipelineMesh final_mesh = untrimmed_mesh;
    if (enable_trim) {
        std::cout << "\n--- [Stage 4] Running Surface Trimming ---\n";
        auto trim_start = std::chrono::steady_clock::now();
        final_mesh = trim_mesh(untrimmed_mesh, trim_opts);
        double trim_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - trim_start).count();
        std::cout << "Trimming finished. Duration: " << trim_duration << " seconds.\n";
        std::cout << "Trimmed mesh has " << final_mesh.vertices.size() << " vertices, " << final_mesh.faces.size() << " faces.\n";

        fs::path s4_out_file = stage4_dir / (stem + "_watertight_trimmed.ply");
        spawn_bg_task(bg_tasks, [s4_out_file, mesh = final_mesh]() {
            std::cout << "Saving trimmed mesh to: " << s4_out_file << "\n";
            if (!save_mesh_ply(s4_out_file, mesh)) {
                std::cerr << "Warning: Failed to save trimmed mesh.\n";
            }
        });
    } else {
        std::cout << "\n--- [Stage 4] Skipping Surface Trimming (Trimming Disabled) ---\n";
        // Write the untrimmed mesh to the trimmed directory to ensure a final mesh always exists at the expected location
        fs::path s4_out_file = stage4_dir / (stem + "_watertight_trimmed.ply");
        spawn_bg_task(bg_tasks, [s4_out_file, mesh = final_mesh]() {
            std::cout << "Saving final watertight (untrimmed) mesh to: " << s4_out_file << "\n";
            if (!save_mesh_ply(s4_out_file, mesh)) {
                std::cerr << "Warning: Failed to save final mesh.\n";
            }
        });
    }

    // -------------------------------------------------------------------------
    // STAGE 5: Skeletonization (Runs on Stage 3 Untrimmed Mesh)
    // -------------------------------------------------------------------------
    std::cout << "\n--- [Stage 5] Running Mesh Skeletonization (on Stage 3 Untrimmed Mesh) ---\n";
    mesh_reconstruction::Triangle_mesh cgal_mesh;
    std::vector<mesh_reconstruction::Triangle_mesh::Vertex_index> vertex_indices;
    vertex_indices.reserve(untrimmed_mesh.vertices.size());
    for (const auto& v : untrimmed_mesh.vertices) {
        vertex_indices.push_back(cgal_mesh.add_vertex(mesh_reconstruction::Point(v.x, v.y, v.z)));
    }
    for (const auto& f : untrimmed_mesh.faces) {
        std::vector<mesh_reconstruction::Triangle_mesh::Vertex_index> face_v;
        face_v.reserve(f.size());
        for (auto idx : f) {
            face_v.push_back(vertex_indices[idx]);
        }
        cgal_mesh.add_face(face_v);
    }

    std::cout << "Normalizing mesh for skeletonization...\n";
    if (!mesh_reconstruction::normalize_mesh_for_skeletonization(cgal_mesh, true)) {
        std::cerr << "Error: mesh normalization failed. Cannot proceed with skeletonization.\n";
        wait_bg_tasks();
        return EXIT_FAILURE;
    }

    // Decimate mesh if faces > 50,000 to keep it efficient
    std::size_t num_faces_before = num_faces(cgal_mesh);
    if (num_faces_before > 50000) {
        std::cout << "Decimating mesh from " << num_faces_before << " faces to ~50000 faces...\n";
        std::size_t target_edges = 75000;
        Stop_predicate stop(target_edges);
        int r = CGAL::Surface_mesh_simplification::edge_collapse(cgal_mesh, stop);
        std::cout << "Decimation finished. Removed " << r << " edges. Current mesh faces: " << num_faces(cgal_mesh) << "\n";

        std::cout << "Re-normalizing decimated mesh...\n";
        if (!mesh_reconstruction::normalize_mesh_for_skeletonization(cgal_mesh, true)) {
            std::cerr << "Error: post-decimation mesh normalization failed. Cannot proceed with skeletonization.\n";
            wait_bg_tasks();
            return EXIT_FAILURE;
        }
    }

    Skeleton skeleton;
    auto skeleton_start = std::chrono::steady_clock::now();
    if (!skeletonize(cgal_mesh, skeleton)) {
        std::cerr << "Error: skeleton extraction failed or produced empty skeleton.\n";
        wait_bg_tasks();
        return EXIT_FAILURE;
    }
    double skeleton_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - skeleton_start).count();
    std::cout << "Skeletonization finished. Duration: " << skeleton_duration << " seconds.\n";

    fs::path s5_out_dir = stage5_dir / stem;
    fs::create_directories(s5_out_dir);
    fs::path s5_out_prefix = s5_out_dir / stem;

    spawn_bg_task(bg_tasks, [s5_out_prefix, skel = std::move(skeleton), mesh = std::move(cgal_mesh)]() {
        std::cout << "Saving skeleton outputs to: " << s5_out_prefix << "\n";
        if (!write_skeleton_outputs(s5_out_prefix.string(), skel, mesh)) {
            std::cerr << "Error: failed to write skeleton outputs.\n";
        }
    });

    auto total_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "\nPipeline stages completed! Total execution time: " << total_duration << " seconds.\n";

    // Wait for all background file write tasks to complete before exiting main
    std::cout << "Waiting for background file writing tasks to complete...\n";
    wait_bg_tasks();
    std::cout << "All background tasks completed. Exiting.\n";

    return EXIT_SUCCESS;
}
