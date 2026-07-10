#include "pipeline_config.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

void print_usage(const char *exe_name) {
  std::cerr
      << "Usage: " << exe_name
      << " [input_ply] [output_dir] [--remove-outliers-percent=VALUE]"
      << " [--outlier-neighbors=K]"
      << " [--enable-smoothing] [--smoothing-neighbors=K]"
      << " [--enable-wlop] [--wlop-retain-percent=VALUE]"
      << " [--wlop-neighbor-radius=VALUE] [--wlop-iterations=N]"
      << " [--wlop-require-uniform-sampling]\n";
}

bool parse_args(const int argc, char *argv[], Pipeline_options &options) {
  if (argc > 1) {
    const std::string first = argv[1];
    if (first == "--help" || first == "-h") {
      print_usage(argv[0]);
      return false;
    }
    options.input_path = first;
  }

  if (argc > 2) {
    options.output_dir = argv[2];
  }

  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];

    if (arg.rfind("--remove-outliers-percent=", 0) == 0) {
      options.outlier_percent = std::stod(arg.substr(26));
      if (options.outlier_percent < 0.0 || options.outlier_percent > 100.0) {
        std::cerr << "Error: --remove-outliers-percent must be in [0, 100].\n";
        return false;
      }
    } else if (arg.rfind("--outlier-neighbors=", 0) == 0) {
      options.outlier_neighbors = std::stoi(arg.substr(20));
      if (options.outlier_neighbors < 2) {
        std::cerr << "Error: --outlier-neighbors must be >= 2.\n";
        return false;
      }
    } else if (arg == "--enable-wlop") {
      options.enable_wlop = true;
    } else if (arg == "--enable-smoothing") {
      options.enable_smoothing = true;
    } else if (arg.rfind("--smoothing-neighbors=", 0) == 0) {
      options.smoothing_neighbors =
          std::stoi(arg.substr(std::string("--smoothing-neighbors=").size()));
      options.enable_smoothing = true;
      if (options.smoothing_neighbors < 2) {
        std::cerr << "Error: --smoothing-neighbors must be >= 2.\n";
        return false;
      }
    } else if (arg.rfind("--wlop-retain-percent=", 0) == 0) {
      const std::string prefix = "--wlop-retain-percent=";
      options.wlop_retain_percent = std::stod(arg.substr(prefix.size()));
      options.enable_wlop = true;
      if (options.wlop_retain_percent <= 0.0 ||
          options.wlop_retain_percent > 100.0) {
        std::cerr << "Error: --wlop-retain-percent must be in (0, 100].\n";
        return false;
      }
    } else if (arg.rfind("--wlop-neighbor-radius=", 0) == 0) {
      const std::string prefix = "--wlop-neighbor-radius=";
      options.wlop_neighbor_radius = std::stod(arg.substr(prefix.size()));
      options.enable_wlop = true;
      if (!std::isfinite(options.wlop_neighbor_radius)) {
        std::cerr << "Error: --wlop-neighbor-radius must be finite.\n";
        return false;
      }
    } else if (arg.rfind("--wlop-iterations=", 0) == 0) {
      const std::string prefix = "--wlop-iterations=";
      const int iterations = std::stoi(arg.substr(prefix.size()));
      options.enable_wlop = true;
      if (iterations < 1) {
        std::cerr << "Error: --wlop-iterations must be >= 1.\n";
        return false;
      }
      options.wlop_iterations = static_cast<unsigned int>(iterations);
    } else if (arg == "--wlop-require-uniform-sampling") {
      options.wlop_require_uniform_sampling = true;
      options.enable_wlop = true;
    } else {
      std::cerr << "Error: unknown option: " << arg << "\n";
      print_usage(argv[0]);
      return false;
    }
  }

  return true;
}

bool prepare_output_dir(const fs::path &out_dir) {
  std::error_code ec;
  fs::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "Error: cannot create output directory '" << out_dir.string()
              << "': " << ec.message() << "\n";
    return false;
  }
  return true;
}

Output_paths make_output_paths(const Pipeline_options &options) {
  const fs::path input(options.input_path);
  const std::string stem = input.stem().string();
  const fs::path out_dir(options.output_dir + "/" + stem);

  Output_paths paths;
  paths.raw_points = out_dir / (stem + "_stage1_raw_points.ply");
  paths.preprocessed_points =
      out_dir / (stem + "_stage1_preprocessed_points.ply");
  return paths;
}
