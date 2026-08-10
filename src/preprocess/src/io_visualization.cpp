#include "io_visualization.h"

#include <fstream>
#include <iostream>

using Point = mesh_reconstruction::Point;
using Vector = mesh_reconstruction::Vector;
using Pwn = mesh_reconstruction::Pwn;

bool write_point_stage_visualization(const std::filesystem::path &out_path,
                                     const mesh_reconstruction::Point_set &points,
                                     const std::string &stage_label) {
  log_stage(stage_label);

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

  std::ofstream out(out_path.string(), std::ios::binary);
  if (!out) {
    std::cerr << "Error: cannot open " << out_path.string()
              << " for writing.\n";
    return false;
  }

  CGAL::IO::set_binary_mode(out);
  out << points;
  if (!out) {
    std::cerr << "Error: failed to write point set to " << out_path.string() << "\n";
    return false;
  }
  out.close();

  std::cout << "Saved: " << out_path.string() << "\n";
  return true;
}

void log_stage(const std::string &label) {
  std::cout << "\n[Stage] " << label << "\n";
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
