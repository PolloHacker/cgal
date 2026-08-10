// SPDX-License-Identifier: MIT
// Part of the C++ port of WNNC (Lin et al., ACM ToG 2024).
#pragma once

#include <algorithm>
#include <cmath>

namespace wnnc {

/// A minimal 3D vector, used for both point positions and normals.
template <typename Scalar>
struct Vec3 {
    Scalar x{};
    Scalar y{};
    Scalar z{};

    constexpr Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    constexpr Vec3& operator*=(Scalar factor) {
        x *= factor;
        y *= factor;
        z *= factor;
        return *this;
    }

    [[nodiscard]] constexpr Scalar dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    [[nodiscard]] constexpr Scalar squaredNorm() const { return dot(*this); }

    [[nodiscard]] Scalar norm() const { return std::sqrt(squaredNorm()); }

    /// Unit-length copy, guarding against division by zero.
    /// (Same epsilon convention as torch.nn.functional.normalize.)
    [[nodiscard]] Vec3 normalized(Scalar epsilon = Scalar(1e-12)) const {
        return *this * (Scalar(1) / std::max(norm(), epsilon));
    }

    template <typename TargetScalar>
    [[nodiscard]] constexpr Vec3<TargetScalar> cast() const {
        return {static_cast<TargetScalar>(x), static_cast<TargetScalar>(y),
                static_cast<TargetScalar>(z)};
    }
};

template <typename Scalar>
[[nodiscard]] constexpr Vec3<Scalar> operator+(Vec3<Scalar> lhs, const Vec3<Scalar>& rhs) {
    return lhs += rhs;
}

template <typename Scalar>
[[nodiscard]] constexpr Vec3<Scalar> operator-(const Vec3<Scalar>& lhs, const Vec3<Scalar>& rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

template <typename Scalar>
[[nodiscard]] constexpr Vec3<Scalar> operator*(Vec3<Scalar> vec, Scalar factor) {
    return vec *= factor;
}

template <typename Scalar>
[[nodiscard]] constexpr Vec3<Scalar> operator*(Scalar factor, Vec3<Scalar> vec) {
    return vec *= factor;
}

}  // namespace wnnc
