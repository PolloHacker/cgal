#include "validation.h"

bool validate_point_set(const std::vector<Pwn> &points, const char *context,
                        const bool require_oriented_normals) {
  if (points.size() < 3) {
    std::cerr << "Error: " << context << " needs at least 3 points.\n";
    return false;
  }

  constexpr double k_min_normal_sq_len = 1e-12;
  for (std::size_t index = 0; index < points.size(); ++index) {
    const Pwn &pwn = points[index];
    if (!std::isfinite(pwn.first.x()) || !std::isfinite(pwn.first.y()) ||
        !std::isfinite(pwn.first.z())) {
      std::cerr << "Error: " << context
                << " contains a non-finite point at index " << index << ".\n";
      return false;
    }

    if (!std::isfinite(pwn.second.x()) || !std::isfinite(pwn.second.y()) ||
        !std::isfinite(pwn.second.z())) {
      std::cerr << "Error: " << context
                << " contains a non-finite normal at index " << index << ".\n";
      return false;
    }

    if (require_oriented_normals &&
        pwn.second.squared_length() <= k_min_normal_sq_len) {
      std::cerr << "Error: " << context
                << " contains a zero-length normal at index " << index << ".\n";
      return false;
    }
  }

  return true;
}

double compute_average_spacing(const std::vector<Pwn> &points,
                               const int neighbors) {
  return CGAL::compute_average_spacing<CGAL::Parallel_tag>(
      points, neighbors,
      CGAL::parameters::point_map(mesh_reconstruction::Point_map()));
}

double compute_average_spacing(const std::vector<Pwn> &points,
                               const int neighbors, const unsigned int max) {
  return CGAL::compute_average_spacing<CGAL::Parallel_tag>(
      points, std::min<unsigned int>(neighbors, 12U),
      CGAL::parameters::point_map(mesh_reconstruction::Point_map()));
}

bool validate_average_spacing(const double average_spacing,
                              const char *context) {
  if (!std::isfinite(average_spacing) || average_spacing <= 0.0) {
    std::cerr << "Error: " << context << " must be finite and positive.\n";
    return false;
  }

  return true;
}