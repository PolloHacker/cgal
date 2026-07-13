// SPDX-License-Identifier: MIT
// Tests for the C++ port of WNNC: the treecode is checked against brute-force
// O(N^2) evaluation of the same kernels, the octree against its structural
// invariants, and the solver end-to-end on a sphere.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <vector>

#include "wnnc/normal_solver.h"
#include "wnnc/octree.h"
#include "wnnc/treecode_kernels.h"
#include "wnnc/vec3.h"
#include "wnnc/winding_number_operator.h"

namespace {

int failureCount = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::printf("FAILED %s:%d: %s\n", __FILE__, __LINE__, #condition);        \
            ++failureCount;                                                           \
        }                                                                             \
    } while (false)

using wnnc::Index;
using Vec3d = wnnc::Vec3<double>;

std::vector<Vec3d> randomPointsInCube(std::mt19937& rng, std::size_t count, double halfWidth) {
    std::uniform_real_distribution<double> coord(-halfWidth, halfWidth);
    std::vector<Vec3d> points(count);
    for (Vec3d& p : points) {
        p = {coord(rng), coord(rng), coord(rng)};
    }
    return points;
}

std::vector<Vec3d> randomUnitVectors(std::mt19937& rng, std::size_t count) {
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::vector<Vec3d> vectors(count);
    for (Vec3d& v : vectors) {
        v = Vec3d{gaussian(rng), gaussian(rng), gaussian(rng)}.normalized();
    }
    return vectors;
}

//// brute-force references ////

std::vector<double> bruteForceA(std::span<const Vec3d> points, std::span<const Vec3d> mu,
                                double width) {
    std::vector<double> result(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        double sum = 0;
        for (std::size_t j = 0; j < points.size(); ++j) {
            sum += wnnc::kernels::windingNumberTerm(points[i] - points[j], mu[j], width);
        }
        result[i] = sum;
    }
    return result;
}

std::vector<Vec3d> bruteForceAT(std::span<const Vec3d> points, std::span<const double> values,
                                double width) {
    std::vector<Vec3d> result(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        Vec3d sum{};
        for (std::size_t j = 0; j < points.size(); ++j) {
            sum += wnnc::kernels::windingNumberTransposeTerm(points[i] - points[j], values[j],
                                                             width);
        }
        result[i] = sum;
    }
    return result;
}

std::vector<Vec3d> bruteForceG(std::span<const Vec3d> points, std::span<const Vec3d> mu,
                               double width) {
    std::vector<Vec3d> result(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        Vec3d sum{};
        for (std::size_t j = 0; j < points.size(); ++j) {
            sum += wnnc::kernels::windingNumberGradientTerm(points[i] - points[j], mu[j], width);
        }
        result[i] = sum;
    }
    return result;
}

double relativeError(std::span<const double> actual, std::span<const double> expected) {
    double diffNormSq = 0;
    double refNormSq = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        diffNormSq += (actual[i] - expected[i]) * (actual[i] - expected[i]);
        refNormSq += expected[i] * expected[i];
    }
    return std::sqrt(diffNormSq / refNormSq);
}

double relativeError(std::span<const Vec3d> actual, std::span<const Vec3d> expected) {
    double diffNormSq = 0;
    double refNormSq = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        diffNormSq += (actual[i] - expected[i]).squaredNorm();
        refNormSq += expected[i].squaredNorm();
    }
    return std::sqrt(diffNormSq / refNormSq);
}

//// tests ////

void testOctreeInvariants(std::mt19937& rng) {
    const std::vector<Vec3d> points = randomPointsInCube(rng, 5000, 0.9);
    const wnnc::Octree<double> tree(points, /*maxDepth=*/15, /*maxPointsPerLeaf=*/1);

    std::vector<int> pointVisits(points.size(), 0);
    for (Index node = 0; node < tree.nodeCount(); ++node) {
        // Pre-order numbering: children have strictly larger indices.
        for (const Index child : tree.children(node)) {
            if (child != wnnc::Octree<double>::kNoChild) {
                CHECK(child > node);
            }
        }
        // Leaves respect the split threshold or the depth cap.
        if (tree.isLeaf(node)) {
            CHECK(tree.pointCount(node) <= 1 || tree.depth() == 15);
            for (const Index point : tree.pointsInNode(node)) {
                ++pointVisits[static_cast<std::size_t>(point)];
            }
        }
    }
    // The leaves partition the point set.
    for (const int visits : pointVisits) {
        CHECK(visits == 1);
    }
    // The root contains everything.
    CHECK(tree.pointCount(0) == static_cast<Index>(points.size()));
}

void testTreecodeMatchesBruteForce(std::mt19937& rng) {
    constexpr std::size_t kPointCount = 3000;
    constexpr double kWidth = 0.01;
    // Far-field approximation error; the treecode is approximate by design.
    constexpr double kTolerance = 0.05;

    const std::vector<Vec3d> points = randomPointsInCube(rng, kPointCount, 0.6);
    const std::vector<Vec3d> mu = randomUnitVectors(rng, kPointCount);
    std::vector<double> values(kPointCount);
    std::uniform_real_distribution<double> valueDist(-1.0, 1.0);
    for (double& value : values) {
        value = valueDist(rng);
    }
    const std::vector<double> widths(kPointCount, kWidth);

    const wnnc::WindingNumberOperator<double> op(points);

    const double errorA = relativeError(op.applyA(mu, widths), bruteForceA(points, mu, kWidth));
    const double errorAT =
        relativeError(op.applyATransposed(values, widths), bruteForceAT(points, values, kWidth));
    const double errorG = relativeError(op.applyG(mu, widths), bruteForceG(points, mu, kWidth));
    std::printf("treecode vs brute force, relative errors: A %.4f, AT %.4f, G %.4f\n", errorA,
                errorAT, errorG);
    CHECK(errorA < kTolerance);
    CHECK(errorAT < kTolerance);
    CHECK(errorG < kTolerance);

    // The root aggregate must be the exact sum of all point attributes.
    const auto aggregates = op.aggregateToNodes(std::span<const Vec3d>(mu));
    Vec3d muSum{};
    for (const Vec3d& m : mu) {
        muSum += m;
    }
    CHECK((aggregates.attrSums[0] - muSum).norm() < 1e-9);
}

void testSphereNormalOrientation(std::mt19937& rng) {
    // Points on a sphere of radius 0.9 (already normalized coordinates);
    // the estimated normals must align with the radial direction and share
    // one global orientation.
    constexpr std::size_t kPointCount = 3000;
    std::vector<Vec3d> points = randomUnitVectors(rng, kPointCount);
    for (Vec3d& p : points) {
        p *= 0.9;
    }

    // float, like the CLI default
    std::vector<wnnc::Vec3<float>> floatPoints;
    floatPoints.reserve(points.size());
    for (const Vec3d& p : points) {
        floatPoints.push_back(p.cast<float>());
    }
    const wnnc::WindingNumberOperator<float> op(std::move(floatPoints));
    // The 'l2' width preset: 3000 random sphere samples have a mean spacing
    // of ~0.06, so the widths must be of comparable scale.
    const wnnc::SolverSettings settings{.iterations = 20, .widthMin = 0.02, .widthMax = 0.08};
    const std::vector<wnnc::Vec3<float>> normals = wnnc::estimateNormals(op, settings);

    double meanRadialAlignment = 0;
    std::size_t consistentCount = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double alignment = normals[i].cast<double>().dot(points[i].normalized());
        meanRadialAlignment += alignment / double(kPointCount);
        consistentCount += alignment > 0;
    }
    std::printf("sphere: mean radial alignment %.4f, outward %zu/%zu\n", meanRadialAlignment,
                consistentCount, points.size());
    // All normals share one global orientation and point along the radius.
    CHECK(std::abs(meanRadialAlignment) > 0.99);
    CHECK(consistentCount == points.size() || consistentCount == 0);
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    testOctreeInvariants(rng);
    testTreecodeMatchesBruteForce(rng);
    testSphereNormalOrientation(rng);

    if (failureCount == 0) {
        std::printf("all tests passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("%d check(s) failed\n", failureCount);
    return EXIT_FAILURE;
}
