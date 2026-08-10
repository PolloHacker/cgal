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
      << " [input_ply] [output_dir]"
      << " [--enable-spatial-subsampling]"
      << " [--spatial-subsample-distance=VALUE] [--min-distance=VALUE]\n";
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

    if (arg == "--enable-spatial-subsampling") {
      options.enable_spatial_subsampling = true;
    } else if (arg.rfind("--spatial-subsample-distance=", 0) == 0) {
      const std::string prefix = "--spatial-subsample-distance=";
      options.spatial_subsample_distance = std::stod(arg.substr(prefix.size()));
      options.enable_spatial_subsampling = true;
      if (options.spatial_subsample_distance <= 0.0) {
        std::cerr << "Error: --spatial-subsample-distance must be > 0.0.\n";
        return false;
      }
    } else if (arg.rfind("--min-distance=", 0) == 0) {
      const std::string prefix = "--min-distance=";
      options.spatial_subsample_distance = std::stod(arg.substr(prefix.size()));
      options.enable_spatial_subsampling = true;
      if (options.spatial_subsample_distance <= 0.0) {
        std::cerr << "Error: --min-distance must be > 0.0.\n";
        return false;
      }
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
