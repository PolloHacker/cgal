// SPDX-License-Identifier: MIT
// Part of the C++ port of WNNC (Lin et al., ACM ToG 2024).
#pragma once

#include <filesystem>
#include <span>
#include <vector>

#include "vec3.h"

namespace wnnc::io {

/// Loads point positions from a .xyz, .obj, .ply or .npy file. Extra columns
/// and attributes (normals, colors, faces, ...) are ignored.
/// Throws std::runtime_error on unsupported or malformed input.
[[nodiscard]] std::vector<Vec3<double>> loadPoints(const std::filesystem::path& path);

/// Writes one "x y z nx ny nz" line per point.
void saveXyzWithNormals(const std::filesystem::path& path, std::span<const Vec3<double>> points,
                        std::span<const Vec3<double>> normals);

}  // namespace wnnc::io
