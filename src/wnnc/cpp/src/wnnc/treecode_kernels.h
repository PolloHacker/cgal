// SPDX-License-Identifier: MIT
// Part of the C++ port of WNNC (Lin et al., ACM ToG 2024).
#pragma once

#include <cmath>

#include "vec3.h"

namespace wnnc::kernels {

// Point-pair terms of the smoothed winding-number operators from the WNNC
// paper. In all three, `diff = x - y` is the vector from the source point y
// to the query point x, and contributions with |diff| < smoothingWidth are
// truncated to zero (the reference implementation's non-continuous kernel).
// `smoothingWidth` must be positive so the singular diff == 0 case is
// truncated rather than divided by zero.

/// A-operator term: the winding-number contribution -(diff . mu) / |diff|^3
/// of a dipole (area-weighted normal) mu located at y, evaluated at x.
template <typename Scalar>
[[nodiscard]] Scalar windingNumberTerm(const Vec3<Scalar>& diff, const Vec3<Scalar>& mu,
                                       Scalar smoothingWidth) {
    const Scalar dist2 = diff.squaredNorm();
    const Scalar dist = std::sqrt(dist2);
    if (dist < smoothingWidth) {
        return Scalar(0);
    }
    return -diff.dot(mu) / (dist * dist2);
}

/// A^T-operator term: diff * value / |diff|^3, the transpose of the term
/// above with a scalar `value` located at y.
template <typename Scalar>
[[nodiscard]] Vec3<Scalar> windingNumberTransposeTerm(const Vec3<Scalar>& diff, Scalar value,
                                                      Scalar smoothingWidth) {
    const Scalar dist2 = diff.squaredNorm();
    const Scalar dist = std::sqrt(dist2);
    if (dist < smoothingWidth) {
        return {};
    }
    return diff * (value / (dist * dist2));
}

/// G-operator term: the (negated) gradient of the winding-number field,
/// mu / |diff|^3 - 3 diff (diff . mu) / |diff|^5.
template <typename Scalar>
[[nodiscard]] Vec3<Scalar> windingNumberGradientTerm(const Vec3<Scalar>& diff,
                                                     const Vec3<Scalar>& mu,
                                                     Scalar smoothingWidth) {
    const Scalar dist2 = diff.squaredNorm();
    const Scalar dist = std::sqrt(dist2);
    if (dist < smoothingWidth) {
        return {};
    }
    const Scalar dist3 = dist * dist2;
    const Scalar dist5 = dist2 * dist3;
    return mu * (Scalar(1) / dist3) - diff * (Scalar(3) * diff.dot(mu) / dist5);
}

}  // namespace wnnc::kernels
