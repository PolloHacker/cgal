// SPDX-License-Identifier: MIT
// Part of the C++ port of WNNC (Lin et al., ACM ToG 2024).
#pragma once

#include <algorithm>
#include <functional>
#include <vector>

#include "vec3.h"
#include "winding_number_operator.h"

namespace wnnc {

struct SolverSettings {
    int iterations = 40;
    /// Smoothing width is annealed linearly from `widthMax` (first iteration)
    /// down to `widthMin` (last iteration).
    double widthMin = 0.002;
    double widthMax = 0.016;
};

/// Called after each iteration with (completedIterations, totalIterations).
using IterationCallback = std::function<void(int, int)>;

/// WNNC normal estimation (the iterative method of Lin et al. 2024, i.e. the
/// main loop of the reference `main_wnnc.py`).
///
/// Starting from mu = 0, each iteration performs:
///  1. A gradient step with exact line search on || A mu - 0.5 ||^2, the
///     requirement that every sample lies on the 0.5 winding-number level
///     set:  r = A^T (0.5 - A mu),  mu += (<r,r> / <A r, A r>) r.
///  2. The WNNC step: replace each mu_i's direction with the direction of
///     the winding-number field gradient (G mu)_i, keeping its length.
///
/// Returns the unit normals of the final iteration.
template <typename Scalar>
std::vector<Vec3<Scalar>> estimateNormals(const WindingNumberOperator<Scalar>& windingNumber,
                                          const SolverSettings& settings,
                                          const IterationCallback& onIteration = {}) {
    const Index pointCount = windingNumber.pointCount();
    const auto n = static_cast<std::size_t>(pointCount);

    // Every sample sits on the surface, where the winding number jumps from
    // 0 (outside) to 1 (inside); its smoothed value there is the target 0.5.
    constexpr Scalar kTargetWindingNumber = Scalar(0.5);

    std::vector<Vec3<Scalar>> mu(n);  // current area-weighted normals
    std::vector<Vec3<Scalar>> unitNormals(n);
    std::vector<Scalar> smoothingWidths(n);

    for (int iteration = 0; iteration < settings.iterations; ++iteration) {
        const double annealing =
            settings.iterations > 1
                ? static_cast<double>(settings.iterations - 1 - iteration) /
                      static_cast<double>(settings.iterations - 1)
                : 0.0;
        const auto width = static_cast<Scalar>(
            settings.widthMin + annealing * (settings.widthMax - settings.widthMin));
        std::fill(smoothingWidths.begin(), smoothingWidths.end(), width);

        // Gradient step with exact line search: r = A^T (0.5 - A mu).
        // (The reference evaluates A^T 0.5 and A^T A mu as two treecode
        // passes; by linearity one pass over the defect is identical, up to
        // the treecode's far-field approximation.)
        std::vector<Scalar> defect = windingNumber.applyA(mu, smoothingWidths);
        for (std::size_t i = 0; i < n; ++i) {
            defect[i] = kTargetWindingNumber - defect[i];
        }
        const std::vector<Vec3<Scalar>> residual =
            windingNumber.applyATransposed(defect, smoothingWidths);
        const std::vector<Scalar> aResidual = windingNumber.applyA(residual, smoothingWidths);

        double residualNormSq = 0.0;
        double aResidualNormSq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            residualNormSq += static_cast<double>(residual[i].squaredNorm());
            aResidualNormSq += static_cast<double>(aResidual[i]) * static_cast<double>(aResidual[i]);
        }
        if (aResidualNormSq == 0.0) {
            break;  // stationary point; avoids a division by zero
        }
        const auto stepSize = static_cast<Scalar>(residualNormSq / aResidualNormSq);
        for (std::size_t i = 0; i < n; ++i) {
            mu[i] += residual[i] * stepSize;
        }

        // WNNC step: new directions from the field gradient, lengths kept.
        const std::vector<Vec3<Scalar>> gradients = windingNumber.applyG(mu, smoothingWidths);
        for (std::size_t i = 0; i < n; ++i) {
            unitNormals[i] = gradients[i].normalized();
            mu[i] = unitNormals[i] * mu[i].norm();
        }

        if (onIteration) {
            onIteration(iteration + 1, settings.iterations);
        }
    }
    return unitNormals;
}

}  // namespace wnnc
