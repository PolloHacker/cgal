#include "io_visualization.h"

#include <fstream>
#include <iostream>

using Point = mesh_reconstruction::Point;
using Vector = mesh_reconstruction::Vector;
using Pwn = mesh_reconstruction::Pwn;

namespace {

bool write_oriented_points_ply(const std::filesystem::path &out_path,
                               const std::vector<Pwn> &points) {

  if (out_path.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(out_path.parent_path(), ec);

    if (ec) {
      std::cerr << "Error: Cannot create directory structure "
                << out_path.parent_path().string()
                << " Reason: " << ec.message() << "\n";
      return false;
    }
  }

  std::ofstream out(out_path.string());
  if (!out) {
    std::cerr << "Error: cannot open " << out_path.string()
              << " for writing.\n";
    return false;
  }

  out << "ply\n";
  out << "format ascii 1.0\n";
  out << "element vertex " << points.size() << "\n";
  out << "property double x\n";
  out << "property double y\n";
  out << "property double z\n";
  out << "property double nx\n";
  out << "property double ny\n";
  out << "property double nz\n";
  out << "end_header\n";

  out.precision(17);
  for (const Pwn &pwn : points) {
    const Point &p = pwn.first;
    const Vector &n = pwn.second;
    out << p.x() << " " << p.y() << " " << p.z() << " " << n.x() << " " << n.y()
        << " " << n.z() << "\n";
  }

  return true;
}

} // namespace

void log_stage(const std::string &label) {
  std::cout << "\n[Stage] " << label << "\n";
}

bool write_point_stage_visualization(const std::filesystem::path &out_path,
                                     const std::vector<Pwn> &points,
                                     const std::string &stage_label) {
  log_stage(stage_label);

  if (!write_oriented_points_ply(out_path, points)) {
    return false;
  }

  std::cout << "Saved: " << out_path.string() << "\n";
  return true;
}

bool write_mesh_stage_visualization(
    const std::filesystem::path &out_path,
    const mesh_reconstruction::Triangle_mesh &mesh,
    const std::string &stage_label) {
  log_stage(stage_label);

  if (!mesh_reconstruction::write_mesh_ply(out_path, mesh)) {
    return false;
  }

  std::cout << "Saved: " << out_path.string() << "\n";
  return true;
}
