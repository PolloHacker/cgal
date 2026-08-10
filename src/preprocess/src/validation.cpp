#include "validation.h"

using Point = mesh_reconstruction::Point;
using Vector = mesh_reconstruction::Vector;

bool validate_point_set(const Point_set &points, const char *context,
                        const bool require_oriented_normals) {
  if (points.size() < 3) {
    std::cerr << "Error: " << context << " needs at least 3 points.\n";
    return false;
  }

  constexpr double k_min_normal_sq_len = 1e-12;
  std::size_t index = 0;
  for (const auto &idx : points) {
    const Point &p = points.point(idx);
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) ||
        !std::isfinite(p.z())) {
      std::cerr << "Error: " << context
                << " contains a non-finite point at index " << index << ".\n";
      return false;
    }

    if (points.has_normal_map()) {
      const Vector &n = points.normal(idx);
      if (!std::isfinite(n.x()) || !std::isfinite(n.y()) ||
          !std::isfinite(n.z())) {
        std::cerr << "Error: " << context
                  << " contains a non-finite normal at index " << index << ".\n";
        return false;
      }

      if (require_oriented_normals &&
          n.squared_length() <= k_min_normal_sq_len) {
        std::cerr << "Error: " << context
                  << " contains a zero-length normal at index " << index << ".\n";
        return false;
      }
    } else if (require_oriented_normals) {
      std::cerr << "Error: " << context
                << " requires oriented normals but has no normal map.\n";
      return false;
    }
    ++index;
  }

  return true;
}
