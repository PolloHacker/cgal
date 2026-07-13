// SPDX-License-Identifier: MIT
// WNNC normal estimation — C++ port of the reference main_wnnc.py
// (Lin, Shi, Liu: "Fast and Globally Consistent Normal Orientation based on
//  the Winding Number Normal Consistency", ACM ToG 2024).
#include <sys/resource.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <CLI11.hpp>

#include "wnnc/normal_solver.h"
#include "wnnc/point_cloud_io.h"
#include "wnnc/vec3.h"
#include "wnnc/winding_number_operator.h"

namespace {

struct Options {
    std::filesystem::path inputPath;
    std::filesystem::path outputDirectory = "results";
    std::string dtype = "float";
    wnnc::SolverSettings solver;
    bool showProgress = false;
};

/// Smoothing-width presets from the WNNC paper, keyed by noise level
/// (l0 = clean uniform samples ... l5 = very noisy or sparse points).
const std::map<std::string, std::pair<double, double>> kPresetWidths = {
    {"l0", {0.002, 0.016}}, {"l1", {0.01, 0.04}}, {"l2", {0.02, 0.08}},
    {"l3", {0.03, 0.12}},   {"l4", {0.04, 0.16}}, {"l5", {0.05, 0.2}},
};

double elapsedSeconds(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

double peakMemoryMegabytes() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);  // bytes
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;  // kilobytes
#endif
}

/// Centers the bounding box at the origin and scales its longest side to
/// 2 / kBoundingBoxScale, so all points fit the octree's [-1, 1]^3 root cube
/// with a safety margin. The smoothing-width presets assume this scale.
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

template <typename Scalar>
std::vector<wnnc::Vec3<double>> estimateNormalsPipeline(
    const std::vector<wnnc::Vec3<double>>& rawPoints, const Options& options) {
    wnnc::WindingNumberOperator<Scalar> windingNumber(normalizeToUnitCube<Scalar>(rawPoints));

    const wnnc::Octree<Scalar>& tree = windingNumber.tree();
    std::cout << "[LOG] octree: " << tree.nodeCount() << " nodes, " << tree.leafCount()
              << " leaves, depth " << tree.depth() << "\n";

    const auto onIteration = [&options](int done, int total) {
        if (!options.showProgress) {
            return;
        }
        std::cerr << "\r[LOG] iteration " << done << "/" << total << std::flush;
        if (done == total) {
            std::cerr << '\n';
        }
    };
    const std::vector<wnnc::Vec3<Scalar>> unitNormals =
        wnnc::estimateNormals(windingNumber, options.solver, onIteration);

    std::vector<wnnc::Vec3<double>> result;
    result.reserve(unitNormals.size());
    for (const wnnc::Vec3<Scalar>& normal : unitNormals) {
        result.push_back(normal.template cast<double>());
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"WNNC normal estimation: globally consistent point-cloud normals "
                 "via winding number normal consistency (Lin et al., ACM ToG 2024)"};

    Options options;
    std::string widthConfig;
    double customWidthMin = 0.04;
    double customWidthMax = 0.01;

    app.add_option("input", options.inputPath,
                   "input point cloud file (.xyz, .obj, .ply or .npy)")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("--width_config", widthConfig,
                   "smoothing width preset l0 (clean) ... l5 (very noisy), or 'custom' "
                   "combined with --wsmin/--wsmax")
        ->required()
        ->check(CLI::IsMember({"l0", "l1", "l2", "l3", "l4", "l5", "custom"}));
    app.add_option("--wsmin", customWidthMin, "custom minimum smoothing width")
        ->check(CLI::PositiveNumber);
    app.add_option("--wsmax", customWidthMax, "custom maximum smoothing width")
        ->check(CLI::PositiveNumber);
    app.add_option("--iters", options.solver.iterations, "number of iterations")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    app.add_option("--out_dir", options.outputDirectory, "output directory")
        ->capture_default_str();
    app.add_option("--dtype", options.dtype,
                   "float is enough for most cases; use double if it is insufficient")
        ->capture_default_str()
        ->check(CLI::IsMember({"float", "double"}));
    app.add_flag("--progress,--tqdm", options.showProgress, "print per-iteration progress");

    CLI11_PARSE(app, argc, argv);

    if (widthConfig == "custom") {
        options.solver.widthMin = customWidthMin;
        options.solver.widthMax = customWidthMax;
    } else {
        std::tie(options.solver.widthMin, options.solver.widthMax) = kPresetWidths.at(widthConfig);
    }
    if (options.solver.widthMin > options.solver.widthMax) {
        std::cerr << "error: wsmin must not exceed wsmax\n";
        return 1;
    }
    std::cout << "[LOG] using width config " << widthConfig << " with wsmin = "
              << options.solver.widthMin << ", wsmax = " << options.solver.widthMax << "\n";

    try {
        const auto timePreprocessStart = std::chrono::steady_clock::now();
        const std::vector<wnnc::Vec3<double>> points = wnnc::io::loadPoints(options.inputPath);
        std::cout << "[LOG] loaded " << points.size() << " points from "
                  << options.inputPath.string() << "\n";

        const auto timeIterationsStart = std::chrono::steady_clock::now();
        const std::vector<wnnc::Vec3<double>> normals =
            options.dtype == "double" ? estimateNormalsPipeline<double>(points, options)
                                      : estimateNormalsPipeline<float>(points, options);
        const double preprocessSeconds =
            std::chrono::duration<double>(timeIterationsStart - timePreprocessStart).count();
        std::cout << "[LOG] time_preproc: " << preprocessSeconds
                  << "\n[LOG] time_main: " << elapsedSeconds(timeIterationsStart) << "\n";

        std::filesystem::create_directories(options.outputDirectory);
        const std::filesystem::path outputPath =
            options.outputDirectory / (options.inputPath.stem().string() + ".xyz");
        wnnc::io::saveXyzWithNormals(outputPath, points, normals);
        std::cout << "[LOG] saved " << outputPath.string() << "\n";
        std::cout << "[LOG] peak mem (MB): " << peakMemoryMegabytes() << "\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
