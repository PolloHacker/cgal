#include "point_cloud_processing.h"

#include "io_visualization.h"
#include "validation.h"

#include <CGAL/IO/read_points.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/jet_smooth_point_set.h>
#include <CGAL/mst_orient_normals.h>
#include <CGAL/remove_outliers.h>
#include <CGAL/wlop_simplify_and_regularize_point_set.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>
#include <unordered_set>
#include <map>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <iomanip>

using Point = mesh_reconstruction::Point;
using Vector = mesh_reconstruction::Vector;
using Point_set = mesh_reconstruction::Point_set;

namespace {

// KD-Tree Adapter for Index Lookup
struct Point_set_property_map {
    typedef Point_set::Index key_type;
    typedef Point value_type;
    typedef const Point& reference;
    typedef boost::lvalue_property_map_tag category;

    const Point_set* ps;
    Point_set_property_map(const Point_set* ps = nullptr) : ps(ps) {}

    reference operator[](key_type idx) const {
        return ps->point(idx);
    }
    friend reference get(const Point_set_property_map& map, key_type idx) {
        return map.ps->point(idx);
    }
};

typedef CGAL::Search_traits_3<mesh_reconstruction::Kernel> Base_traits;
typedef CGAL::Search_traits_adapter<Point_set::Index, Point_set_property_map, Base_traits> Traits;
typedef CGAL::Distance_adapter<Point_set::Index, Point_set_property_map, CGAL::Euclidean_distance<Base_traits>> Distance;
typedef CGAL::Orthogonal_k_neighbor_search<Traits, Distance> Neighbor_search;
typedef Neighbor_search::Tree Search_tree;

enum class PlyType {
    INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64, UNKNOWN
};

inline PlyType string_to_ply_type(const std::string& type_str) {
    if (type_str == "char" || type_str == "int8") return PlyType::INT8;
    if (type_str == "uchar" || type_str == "uint8") return PlyType::UINT8;
    if (type_str == "short" || type_str == "int16") return PlyType::INT16;
    if (type_str == "ushort" || type_str == "uint16") return PlyType::UINT16;
    if (type_str == "int" || type_str == "int32") return PlyType::INT32;
    if (type_str == "uint" || type_str == "uint32") return PlyType::UINT32;
    if (type_str == "float" || type_str == "float32") return PlyType::FLOAT32;
    if (type_str == "double" || type_str == "float64") return PlyType::FLOAT64;
    return PlyType::UNKNOWN;
}

inline std::size_t ply_type_size(PlyType type) {
    switch (type) {
        case PlyType::INT8:
        case PlyType::UINT8: return 1;
        case PlyType::INT16:
        case PlyType::UINT16: return 2;
        case PlyType::INT32:
        case PlyType::UINT32:
        case PlyType::FLOAT32: return 4;
        case PlyType::FLOAT64: return 8;
        default: return 0;
    }
}

struct PlyProperty {
    std::string name;
    std::string type_str;
    PlyType type;
    std::size_t size;
    std::size_t offset;
};

struct PlyHeader {
    enum class Format { ASCII, BINARY_LE, BINARY_BE, UNSUPPORTED } format = Format::UNSUPPORTED;
    std::size_t vertex_count = 0;
    std::vector<PlyProperty> vertex_properties;
    std::size_t vertex_byte_size = 0;
    
    int idx_x = -1, idx_y = -1, idx_z = -1;
    int idx_nx = -1, idx_ny = -1, idx_nz = -1;
    int idx_red = -1, idx_green = -1, idx_blue = -1;
    int idx_intensity = -1;
};

struct VoxelKey {
    int64_t x, y, z;
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const {
        std::size_t h = 0;
        auto hash_combine = [&h](uint64_t val) {
            h ^= std::hash<uint64_t>{}(val) + 0x9e3779b9 + (h << 6) + (h >> 2);
        };
        hash_combine(static_cast<uint64_t>(k.x));
        hash_combine(static_cast<uint64_t>(k.y));
        hash_combine(static_cast<uint64_t>(k.z));
        return h;
    }
};

inline bool get_line_safe(std::istream& is, std::string& line) {
    line.clear();
    char c;
    while (is.get(c)) {
        if (c == '\n') {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return true;
        }
        line.push_back(c);
    }
    return !line.empty();
}

inline double read_binary_value(const char* ptr, PlyType type) {
    switch (type) {
        case PlyType::INT8: return static_cast<double>(*reinterpret_cast<const int8_t*>(ptr));
        case PlyType::UINT8: return static_cast<double>(*reinterpret_cast<const uint8_t*>(ptr));
        case PlyType::INT16: {
            int16_t val;
            std::memcpy(&val, ptr, 2);
            return static_cast<double>(val);
        }
        case PlyType::UINT16: {
            uint16_t val;
            std::memcpy(&val, ptr, 2);
            return static_cast<double>(val);
        }
        case PlyType::INT32: {
            int32_t val;
            std::memcpy(&val, ptr, 4);
            return static_cast<double>(val);
        }
        case PlyType::UINT32: {
            uint32_t val;
            std::memcpy(&val, ptr, 4);
            return static_cast<double>(val);
        }
        case PlyType::FLOAT32: {
            float val;
            std::memcpy(&val, ptr, 4);
            return static_cast<double>(val);
        }
        case PlyType::FLOAT64: {
            double val;
            std::memcpy(&val, ptr, 8);
            return val;
        }
        default: return 0.0;
    }
}

inline std::vector<std::string> tokenize(const std::string& str) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : str) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        } else {
            token.push_back(c);
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

inline double parse_ascii_value(const std::string& token, PlyType type) {
    if (type == PlyType::FLOAT32 || type == PlyType::FLOAT64) {
        return std::stod(token);
    } else {
        return static_cast<double>(std::stoll(token));
    }
}

inline unsigned char to_uchar(double val, PlyType type) {
    if (type == PlyType::FLOAT32 || type == PlyType::FLOAT64) {
        if (val >= 0.0 && val <= 1.0) {
            return static_cast<unsigned char>(std::round(val * 255.0));
        }
    }
    return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
}

bool parse_ply_header(const std::string& filepath, PlyHeader& header, std::ifstream& file) {
    file.open(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: cannot open file " << filepath << "\n";
        return false;
    }
    
    std::string line;
    if (!get_line_safe(file, line) || line != "ply") {
        std::cerr << "Error: " << filepath << " is not a valid PLY file (missing 'ply' magic)\n";
        return false;
    }
    
    std::string current_element = "";
    bool reading_vertex_properties = false;
    
    while (get_line_safe(file, line)) {
        std::stringstream ss(line);
        std::string token;
        ss >> token;
        
        if (token == "format") {
            std::string format_type, version;
            ss >> format_type >> version;
            if (format_type == "ascii") {
                header.format = PlyHeader::Format::ASCII;
            } else if (format_type == "binary_little_endian") {
                header.format = PlyHeader::Format::BINARY_LE;
            } else if (format_type == "binary_big_endian") {
                header.format = PlyHeader::Format::BINARY_BE;
            } else {
                header.format = PlyHeader::Format::UNSUPPORTED;
            }
        } else if (token == "element") {
            std::string element_name;
            std::size_t count;
            ss >> element_name >> count;
            current_element = element_name;
            if (element_name == "vertex") {
                header.vertex_count = count;
                reading_vertex_properties = true;
            } else {
                reading_vertex_properties = false;
            }
        } else if (token == "property") {
            if (reading_vertex_properties) {
                std::string type_str, name_str;
                ss >> type_str;
                if (type_str == "list") {
                    std::cerr << "Error: list property in vertex element is not supported.\n";
                    return false;
                }
                ss >> name_str;
                PlyType type = string_to_ply_type(type_str);
                if (type == PlyType::UNKNOWN) {
                    std::cerr << "Error: unknown property type '" << type_str << "' in PLY header.\n";
                    return false;
                }
                
                std::size_t size = ply_type_size(type);
                std::size_t offset = header.vertex_byte_size;
                header.vertex_byte_size += size;
                
                PlyProperty prop = { name_str, type_str, type, size, offset };
                header.vertex_properties.push_back(prop);
            }
        } else if (token == "end_header") {
            break;
        }
    }
    
    // Map property indices
    for (int i = 0; i < static_cast<int>(header.vertex_properties.size()); ++i) {
        const auto& prop = header.vertex_properties[i];
        std::string name = prop.name;
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
        
        if (lower_name == "x") header.idx_x = i;
        else if (lower_name == "y") header.idx_y = i;
        else if (lower_name == "z") header.idx_z = i;
        else if (lower_name == "nx" || lower_name == "normal_x") header.idx_nx = i;
        else if (lower_name == "ny" || lower_name == "normal_y") header.idx_ny = i;
        else if (lower_name == "nz" || lower_name == "normal_z") header.idx_nz = i;
        else if (lower_name == "red" || lower_name == "r") header.idx_red = i;
        else if (lower_name == "green" || lower_name == "g") header.idx_green = i;
        else if (lower_name == "blue" || lower_name == "b") header.idx_blue = i;
        else if (lower_name.find("intensity") != std::string::npos) header.idx_intensity = i;
    }
    
    if (header.idx_x == -1 || header.idx_y == -1 || header.idx_z == -1) {
        std::cerr << "Error: PLY file does not contain x, y, z properties.\n";
        return false;
    }
    
    if (header.format == PlyHeader::Format::UNSUPPORTED) {
        std::cerr << "Error: unsupported PLY format.\n";
        return false;
    }
    
    if (header.format == PlyHeader::Format::BINARY_BE) {
        std::cerr << "Error: binary_big_endian format is not supported.\n";
        return false;
    }
    
    return true;
}

bool load_oriented_points_spatial(const std::string &input_path,
                                  Point_set &points,
                                  double min_distance) {
    PlyHeader header;
    std::ifstream file;
    if (!parse_ply_header(input_path, header, file)) {
        return false;
    }
    
    std::cout << "PLY Metadata parsed successfully:\n";
    std::cout << "  Format:       " << (header.format == PlyHeader::Format::ASCII ? "ASCII" : "Binary (Little Endian)") << "\n";
    std::cout << "  Point Count:  " << header.vertex_count << "\n";
    
    const bool has_normals = (header.idx_nx != -1 && header.idx_ny != -1 && header.idx_nz != -1);
    points.add_normal_map(); // We always guarantee normal map exists for pipeline consistency.
    
    const bool has_colors = (header.idx_red != -1 && header.idx_green != -1 && header.idx_blue != -1);
    Point_set::Property_map<unsigned char> red_map;
    Point_set::Property_map<unsigned char> green_map;
    Point_set::Property_map<unsigned char> blue_map;
    if (has_colors) {
        red_map = points.add_property_map<unsigned char>("red", 0).first;
        green_map = points.add_property_map<unsigned char>("green", 0).first;
        blue_map = points.add_property_map<unsigned char>("blue", 0).first;
    }
    
    const bool has_intensity = (header.idx_intensity != -1);
    std::string intensity_prop_name = "";
    bool intensity_is_double = false;
    Point_set::Property_map<float> intensity_map_float;
    Point_set::Property_map<double> intensity_map_double;
    if (has_intensity) {
        intensity_prop_name = header.vertex_properties[header.idx_intensity].name;
        intensity_is_double = (header.vertex_properties[header.idx_intensity].type == PlyType::FLOAT64);
        if (intensity_is_double) {
            intensity_map_double = points.add_property_map<double>(intensity_prop_name, 0.0).first;
        } else {
            intensity_map_float = points.add_property_map<float>(intensity_prop_name, 0.0f).first;
        }
    }
    
    std::unordered_set<VoxelKey, VoxelKeyHash> voxel_grid;
    
    double orig_min_x = std::numeric_limits<double>::infinity();
    double orig_min_y = std::numeric_limits<double>::infinity();
    double orig_min_z = std::numeric_limits<double>::infinity();
    double orig_max_x = -std::numeric_limits<double>::infinity();
    double orig_max_y = -std::numeric_limits<double>::infinity();
    double orig_max_z = -std::numeric_limits<double>::infinity();
    
    std::vector<char> record_buf(header.vertex_byte_size);
    std::string ascii_line;
    
    for (std::size_t i = 0; i < header.vertex_count; ++i) {
        double x = 0.0, y = 0.0, z = 0.0;
        double nx = 0.0, ny = 0.0, nz = 0.0;
        double r = 0.0, g = 0.0, b = 0.0;
        double intensity = 0.0;
        
        if (header.format == PlyHeader::Format::BINARY_LE) {
            if (!file.read(record_buf.data(), header.vertex_byte_size)) {
                break;
            }
            x = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_x].offset, header.vertex_properties[header.idx_x].type);
            y = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_y].offset, header.vertex_properties[header.idx_y].type);
            z = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_z].offset, header.vertex_properties[header.idx_z].type);
        } else {
            if (!get_line_safe(file, ascii_line)) {
                break;
            }
            std::vector<std::string> tokens = tokenize(ascii_line);
            if (tokens.size() < header.vertex_properties.size()) {
                continue;
            }
            x = parse_ascii_value(tokens[header.idx_x], header.vertex_properties[header.idx_x].type);
            y = parse_ascii_value(tokens[header.idx_y], header.vertex_properties[header.idx_y].type);
            z = parse_ascii_value(tokens[header.idx_z], header.vertex_properties[header.idx_z].type);
        }
        
        orig_min_x = std::min(orig_min_x, x);
        orig_min_y = std::min(orig_min_y, y);
        orig_min_z = std::min(orig_min_z, z);
        orig_max_x = std::max(orig_max_x, x);
        orig_max_y = std::max(orig_max_y, y);
        orig_max_z = std::max(orig_max_z, z);
        
        int64_t gx = static_cast<int64_t>(std::floor(x / min_distance));
        int64_t gy = static_cast<int64_t>(std::floor(y / min_distance));
        int64_t gz = static_cast<int64_t>(std::floor(z / min_distance));
        VoxelKey key = { gx, gy, gz };
        
        if (voxel_grid.insert(key).second) {
            Point p(x, y, z);
            Point_set::Index idx = *(points.insert(p));
            
            if (header.format == PlyHeader::Format::BINARY_LE) {
                if (has_normals) {
                    nx = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_nx].offset, header.vertex_properties[header.idx_nx].type);
                    ny = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_ny].offset, header.vertex_properties[header.idx_ny].type);
                    nz = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_nz].offset, header.vertex_properties[header.idx_nz].type);
                }
                if (has_colors) {
                    r = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_red].offset, header.vertex_properties[header.idx_red].type);
                    g = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_green].offset, header.vertex_properties[header.idx_green].type);
                    b = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_blue].offset, header.vertex_properties[header.idx_blue].type);
                }
                if (has_intensity) {
                    intensity = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_intensity].offset, header.vertex_properties[header.idx_intensity].type);
                }
            } else {
                std::vector<std::string> tokens = tokenize(ascii_line);
                if (has_normals) {
                    nx = parse_ascii_value(tokens[header.idx_nx], header.vertex_properties[header.idx_nx].type);
                    ny = parse_ascii_value(tokens[header.idx_ny], header.vertex_properties[header.idx_ny].type);
                    nz = parse_ascii_value(tokens[header.idx_nz], header.vertex_properties[header.idx_nz].type);
                }
                if (has_colors) {
                    r = parse_ascii_value(tokens[header.idx_red], header.vertex_properties[header.idx_red].type);
                    g = parse_ascii_value(tokens[header.idx_green], header.vertex_properties[header.idx_green].type);
                    b = parse_ascii_value(tokens[header.idx_blue], header.vertex_properties[header.idx_blue].type);
                }
                if (has_intensity) {
                    intensity = parse_ascii_value(tokens[header.idx_intensity], header.vertex_properties[header.idx_intensity].type);
                }
            }
            
            points.normal(idx) = Vector(nx, ny, nz);
            
            if (has_colors) {
                red_map[idx] = to_uchar(r, header.vertex_properties[header.idx_red].type);
                green_map[idx] = to_uchar(g, header.vertex_properties[header.idx_green].type);
                blue_map[idx] = to_uchar(b, header.vertex_properties[header.idx_blue].type);
            }
            if (has_intensity) {
                if (intensity_is_double) {
                    intensity_map_double[idx] = intensity;
                } else {
                    intensity_map_float[idx] = static_cast<float>(intensity);
                }
            }
        }
        
        if ((i + 1) % 10000000 == 0 || i + 1 == header.vertex_count) {
            double progress_pct = 100.0 * (i + 1) / header.vertex_count;
            std::cout << "Parsed: " << (i + 1) << " / " << header.vertex_count 
                      << " (" << std::fixed << std::setprecision(1) << progress_pct << "%), "
                      << "Decimated: " << points.size() << " points\n";
        }
    }
    
    file.close();
    
    std::size_t decimated_count = points.size();
    double reduction_pct = 100.0 * (1.0 - (double)decimated_count / header.vertex_count);
    
    std::cout << "--------------------------------------------------\n";
    std::cout << "Streaming Phase Completed:\n";
    std::cout << "  Decimated Point Count: " << decimated_count << " / " << header.vertex_count << "\n";
    std::cout << "  Reduction Percentage:  " << std::fixed << std::setprecision(2) << reduction_pct << "%\n";
    std::cout << "--------------------------------------------------\n";
    
    points.collect_garbage();
    return true;
}

bool smooth_points(Point_set &points, const int requested_neighbors) {
  if (!validate_point_set(points, "smoothing input", false)) {
    return false;
  }

  const std::size_t max_neighbors = points.size() - 1;
  const unsigned int neighbors =
      static_cast<unsigned int>(std::min<std::size_t>(
          static_cast<std::size_t>(requested_neighbors), max_neighbors));

  if (neighbors < 2) {
    std::cerr << "Error: smoothing requires at least 2 neighbors.\n";
    return false;
  }

  CGAL::jet_smooth_point_set<CGAL::Parallel_tag>(points, neighbors);

  std::cout << "Smoothing neighbors: " << neighbors << "\n";
  return validate_point_set(points, "post-smoothing point set", false);
}

bool apply_wlop_downsampling(Point_set &points,
                             const Pipeline_options &options) {
  if (!validate_point_set(points, "WLOP input", false)) {
    return false;
  }

  const std::size_t input_count = points.size();
  std::vector<Point> downsampled_points;
  downsampled_points.reserve(
      std::max<std::size_t>(3, static_cast<std::size_t>(std::ceil(
                                   static_cast<double>(input_count) *
                                   (options.wlop_retain_percent / 100.0)))));

  const double wlop_radius =
      (options.wlop_neighbor_radius <= 0.0)
          ? -1.0
          : options.wlop_neighbor_radius *
                compute_average_spacing(points, options.outlier_neighbors);

  CGAL::wlop_simplify_and_regularize_point_set<CGAL::Parallel_tag>(
      points, std::back_inserter(downsampled_points),
      CGAL::parameters::point_map(points.point_map())
          .select_percentage(options.wlop_retain_percent)
          .neighbor_radius(wlop_radius)
          .number_of_iterations(options.wlop_iterations)
          .require_uniform_sampling(options.wlop_require_uniform_sampling));

  if (downsampled_points.size() < 3) {
    std::cerr << "Error: WLOP produced too few points ("
              << downsampled_points.size()
              << "). Increase --wlop-retain-percent.\n";
    return false;
  }

  // Build KD-tree on original points to transfer attributes
  std::cout << "WLOP complete. Building 1-NN KD-tree to transfer attributes...\n";
  
  const bool has_normals = points.has_normal_map();
  
  auto red_prop = points.property_map<unsigned char>("red");
  bool has_colors = red_prop.second;
  
  auto intensity_float_prop = points.property_map<float>("intensity");
  auto intensity_double_prop = points.property_map<double>("intensity");
  bool has_intensity_float = intensity_float_prop.second;
  bool has_intensity_double = intensity_double_prop.second;

  // KD-tree is built using the namespace-scope adapter types defined above.

  Point_set_property_map ppmap(&points);
  Traits traits(ppmap);
  Distance distance(ppmap);
  Search_tree search_tree(points.begin(), points.end(), Search_tree::Splitter(), traits);

  // Setup final Point Set
  Point_set final_points;
  if (has_normals) final_points.add_normal_map();
  
  Point_set::Property_map<unsigned char> f_red_map, f_green_map, f_blue_map;
  Point_set::Property_map<unsigned char> red_map, green_map, blue_map;
  if (has_colors) {
    f_red_map = final_points.add_property_map<unsigned char>("red", 0).first;
    f_green_map = final_points.add_property_map<unsigned char>("green", 0).first;
    f_blue_map = final_points.add_property_map<unsigned char>("blue", 0).first;
    
    red_map = red_prop.first;
    green_map = points.property_map<unsigned char>("green").first;
    blue_map = points.property_map<unsigned char>("blue").first;
  }
  
  Point_set::Property_map<float> f_intensity_map_float;
  Point_set::Property_map<double> f_intensity_map_double;
  if (has_intensity_float) {
    f_intensity_map_float = final_points.add_property_map<float>("intensity", 0.0f).first;
  }
  if (has_intensity_double) {
    f_intensity_map_double = final_points.add_property_map<double>("intensity", 0.0).first;
  }

  // Perform 1-NN query for each WLOP point
  for (const auto& w_pt : downsampled_points) {
    Neighbor_search search(search_tree, w_pt, 1, 0.0, true, distance);
    if (search.begin() != search.end()) {
      auto old_idx = search.begin()->first;
      Point_set::Index new_idx = *(final_points.insert(w_pt));
      
      if (has_normals) {
        final_points.normal(new_idx) = points.normal(old_idx);
      }
      if (has_colors) {
        f_red_map[new_idx] = red_map[old_idx];
        f_green_map[new_idx] = green_map[old_idx];
        f_blue_map[new_idx] = blue_map[old_idx];
      }
      if (has_intensity_float) {
        f_intensity_map_float[new_idx] = intensity_float_prop.first[old_idx];
      }
      if (has_intensity_double) {
        f_intensity_map_double[new_idx] = intensity_double_prop.first[old_idx];
      }
    }
  }

  points = std::move(final_points);

  std::cout << "WLOP retain percent: " << options.wlop_retain_percent << "\n";
  std::cout << "WLOP neighbor radius: " << options.wlop_neighbor_radius
            << (options.wlop_neighbor_radius <= 0.0 ? " (auto)" : "") << "\n";
  std::cout << "WLOP iterations: " << options.wlop_iterations << "\n";
  std::cout << "WLOP uniform sampling: "
            << (options.wlop_require_uniform_sampling ? "enabled" : "disabled")
            << "\n";
  std::cout << "WLOP points: " << input_count << " -> " << points.size()
            << "\n";

  return true;
}

} // namespace

bool load_oriented_points(const std::string &input_path,
                          Point_set &points,
                          const Pipeline_options &options) {
  if (options.enable_spatial_subsampling) {
    log_stage("1. Load point cloud with Spatial Subsampling");
    if (!load_oriented_points_spatial(input_path, points, options.spatial_subsample_distance)) {
      return false;
    }
  } else {
    log_stage("1. Load point cloud + normals (PLY)");
    if (!CGAL::IO::read_point_set(input_path, points)) {
      std::cerr << "Error: cannot read point set from " << input_path << "\n";
      return false;
    }

    if (!points.has_normal_map()) {
      points.add_normal_map();
      for (const auto &idx : points) {
        points.normal(idx) = Vector(0.0, 0.0, 0.0);
      }
      std::cout << "Input normals were not found/readable. Falling back to zero normals.\n";
    }
  }

  if (!validate_point_set(points, "loaded point set", false)) {
    return false;
  }

  std::cout << "Loaded points: " << points.size() << "\n";
  return true;
}

bool preprocess_points(Point_set &points, const Pipeline_options &options,
                       double &average_spacing) {
  if (!validate_point_set(points, "loaded point set", false)) {
    return false;
  }

  if (options.outlier_percent > 0.0) {
    log_stage("1.2 Outlier removal");
    average_spacing = compute_average_spacing(points, options.outlier_neighbors);

    auto rout_it = CGAL::remove_outliers<CGAL::Parallel_tag>(
        points, 
        options.outlier_neighbors, 
        points.parameters().threshold_percent(options.outlier_percent)
                           .threshold_distance(average_spacing)
    );
    const std::size_t before = points.size();
    points.remove(rout_it, points.end());
    points.collect_garbage();
    std::cout << "Outlier removal: " << (before - points.size())
              << " points removed.\n";
  } else {
    std::cout << "Outlier removal: disabled.\n";
  }

  if (!validate_point_set(points, "post-outlier point set", false)) {
    return false;
  }

  if (options.enable_wlop) {
    log_stage("1.3 WLOP downsampling");
    if (options.wlop_retain_percent < 10.0) {
      std::cout << "Warning: WLOP retain percent is very low and may "
                   "destabilize Poisson reconstruction.\n";
    }
    if (!apply_wlop_downsampling(points, options)) {
      return false;
    }
  } else {
    std::cout << "WLOP downsampling: disabled.\n";
  }

  if (!validate_point_set(points, "post-WLOP point set", false)) {
    return false;
  }

  if (options.enable_smoothing) {
    log_stage("1.4 Jet smoothing");
    if (!smooth_points(points, options.smoothing_neighbors)) {
      return false;
    }
  } else {
    std::cout << "Smoothing: disabled.\n";
  }

  if (!validate_point_set(points, "preprocessed point set", false)) {
    return false;
  }

  log_stage("1.6 Average spacing");
  average_spacing = compute_average_spacing(points, options.outlier_neighbors);

  if (!validate_average_spacing(average_spacing, "average spacing")) {
    return false;
  }

  std::cout << "Average spacing (k=" << options.outlier_neighbors << "): " << average_spacing << "\n";
  return true;
}
