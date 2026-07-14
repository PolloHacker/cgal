#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <string_view>
#include <charconv>
#include <sstream>
#include <memory>
#include <cmath>
#include <unordered_set>
#include <cstring>
#include <algorithm>
#include <limits>
#include <chrono>
#include <iomanip>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/wlop_simplify_and_regularize_point_set.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

// Types
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Vector_3 Vector;
typedef CGAL::Point_set_3<Point> Point_set;

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

// Safe line-reading to handle Windows/Linux newlines
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

// Fast zero-allocation tokenizer for ASCII files
inline void tokenize_to_views(std::string_view str, std::vector<std::string_view>& tokens) {
    tokens.clear();
    size_t start = 0;
    while (start < str.size()) {
        // Skip leading whitespace
        while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start]))) {
            ++start;
        }
        if (start >= str.size()) {
            break;
        }
        size_t end = start;
        while (end < str.size() && !std::isspace(static_cast<unsigned char>(str[end]))) {
            ++end;
        }
        tokens.push_back(str.substr(start, end - start));
        start = end;
    }
}

inline double parse_ascii_view(std::string_view sv, PlyType type) {
    if (type == PlyType::FLOAT32 || type == PlyType::FLOAT64) {
        double val = 0.0;
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
        if (ec == std::errc{}) {
            return val;
        }
        return std::stod(std::string(sv));
    } else {
        long long val = 0;
        auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), val);
        if (ec == std::errc{}) {
            return static_cast<double>(val);
        }
        return static_cast<double>(std::stoll(std::string(sv)));
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

typedef CGAL::Search_traits_3<Kernel> Base_traits;
typedef CGAL::Search_traits_adapter<Point_set::Index, Point_set_property_map, Base_traits> Traits;
typedef CGAL::Distance_adapter<Point_set::Index, Point_set_property_map, CGAL::Euclidean_distance<Base_traits>> Distance;
typedef CGAL::Orthogonal_k_neighbor_search<Traits, Distance> Neighbor_search;
typedef Neighbor_search::Tree Search_tree;

void print_usage(const char* exe_name) {
    std::cout << "Usage: " << exe_name << " <input_path.ply> <output_path.ply> [options]\n"
              << "Options:\n"
              << "  --min-distance <float>        Voxel edge length (default: 0.1)\n"
              << "  --wlop-retain-percent <float> WLOP retain percent (default: 10.0)\n"
              << "  --wlop-iterations <int>       WLOP iterations (default: 35)\n"
              << "  --wlop-require-uniform        Enable uniform sampling in WLOP\n"
              << "  -h, --help                    Show this help message\n";
}

int main(int argc, char* argv[]) {
    auto start_time = std::chrono::high_resolution_clock::now();

    if (argc < 3) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    
    std::string input_path = argv[1];
    std::string output_path = argv[2];
    
    if (input_path == "--help" || input_path == "-h" || output_path == "--help" || output_path == "-h") {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    
    double min_distance = 0.1;
    double wlop_retain_percent = 10.0;
    unsigned int wlop_iterations = 35;
    bool wlop_require_uniform = false;
    
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--min-distance" && i + 1 < argc) {
            min_distance = std::stod(argv[++i]);
        } else if (arg == "--wlop-retain-percent" && i + 1 < argc) {
            wlop_retain_percent = std::stod(argv[++i]);
        } else if (arg == "--wlop-iterations" && i + 1 < argc) {
            wlop_iterations = std::stoi(argv[++i]);
        } else if (arg == "--wlop-require-uniform") {
            wlop_require_uniform = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::cerr << "Error: unknown argument " << arg << "\n";
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }
    
    if (min_distance <= 0.0) {
        std::cerr << "Error: --min-distance must be strictly positive.\n";
        return EXIT_FAILURE;
    }
    
    std::cout << "--------------------------------------------------\n";
    std::cout << "Streamed Spatial Subsampler Configuration:\n";
    std::cout << "  Input Path:            " << input_path << "\n";
    std::cout << "  Output Path:           " << output_path << "\n";
    std::cout << "  Min Distance:          " << min_distance << "\n";
    std::cout << "  WLOP Retain Percent:   " << wlop_retain_percent << "%\n";
    std::cout << "  WLOP Iterations:       " << wlop_iterations << "\n";
    std::cout << "  WLOP Uniform Sampling: " << (wlop_require_uniform ? "Yes" : "No") << "\n";
    std::cout << "--------------------------------------------------\n";
    
    // Parse header
    PlyHeader header;
    std::ifstream file;
    if (!parse_ply_header(input_path, header, file)) {
        return EXIT_FAILURE;
    }
    
    std::cout << "PLY Metadata parsed successfully:\n";
    std::cout << "  Format:       " << (header.format == PlyHeader::Format::ASCII ? "ASCII" : "Binary (Little Endian)") << "\n";
    std::cout << "  Point Count:  " << header.vertex_count << "\n";
    std::cout << "  Vertex Size:  " << header.vertex_byte_size << " bytes\n";
    std::cout << "  Normals:      " << (header.idx_nx != -1 ? "Yes" : "No") << "\n";
    std::cout << "  Colors:       " << (header.idx_red != -1 ? "Yes" : "No") << "\n";
    std::cout << "  Intensity:    " << (header.idx_intensity != -1 ? "Yes (" + header.vertex_properties[header.idx_intensity].name + ")" : "No") << "\n";
    std::cout << "--------------------------------------------------\n";

    // Setup Point_set_3 and dynamic properties
    Point_set point_set;
    
    // Normals property
    const bool has_normals = (header.idx_nx != -1 && header.idx_ny != -1 && header.idx_nz != -1);
    if (has_normals) {
        point_set.add_normal_map();
    }
    
    // Color properties
    const bool has_colors = (header.idx_red != -1 && header.idx_green != -1 && header.idx_blue != -1);
    Point_set::Property_map<unsigned char> red_map;
    Point_set::Property_map<unsigned char> green_map;
    Point_set::Property_map<unsigned char> blue_map;
    if (has_colors) {
        red_map = point_set.add_property_map<unsigned char>("red", 0).first;
        green_map = point_set.add_property_map<unsigned char>("green", 0).first;
        blue_map = point_set.add_property_map<unsigned char>("blue", 0).first;
    }
    
    // Intensity property
    const bool has_intensity = (header.idx_intensity != -1);
    std::string intensity_prop_name = "";
    bool intensity_is_double = false;
    Point_set::Property_map<float> intensity_map_float;
    Point_set::Property_map<double> intensity_map_double;
    if (has_intensity) {
        intensity_prop_name = header.vertex_properties[header.idx_intensity].name;
        intensity_is_double = (header.vertex_properties[header.idx_intensity].type == PlyType::FLOAT64);
        if (intensity_is_double) {
            intensity_map_double = point_set.add_property_map<double>(intensity_prop_name, 0.0).first;
        } else {
            intensity_map_float = point_set.add_property_map<float>(intensity_prop_name, 0.0f).first;
        }
    }
    
    // Voxel grid for bucketing
    std::unordered_set<VoxelKey, VoxelKeyHash> voxel_grid;
    
    // Original bounding box calculation on-the-fly
    double orig_min_x = std::numeric_limits<double>::infinity();
    double orig_min_y = std::numeric_limits<double>::infinity();
    double orig_min_z = std::numeric_limits<double>::infinity();
    double orig_max_x = -std::numeric_limits<double>::infinity();
    double orig_max_y = -std::numeric_limits<double>::infinity();
    double orig_max_z = -std::numeric_limits<double>::infinity();
    
    std::cout << "Streaming and Spatial Filtering Loop started...\n";
    
    std::vector<char> record_buf(header.vertex_byte_size);
    std::string ascii_line;
    std::vector<std::string_view> tokens;
    
    for (std::size_t i = 0; i < header.vertex_count; ++i) {
        double x = 0.0, y = 0.0, z = 0.0;
        double nx = 0.0, ny = 0.0, nz = 0.0;
        double r = 0.0, g = 0.0, b = 0.0;
        double intensity = 0.0;
        
        if (header.format == PlyHeader::Format::BINARY_LE) {
            if (!file.read(record_buf.data(), header.vertex_byte_size)) {
                std::cerr << "Warning: Read failed or reached unexpected EOF at record " << i << "\n";
                break;
            }
            
            x = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_x].offset, header.vertex_properties[header.idx_x].type);
            y = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_y].offset, header.vertex_properties[header.idx_y].type);
            z = read_binary_value(record_buf.data() + header.vertex_properties[header.idx_z].offset, header.vertex_properties[header.idx_z].type);
        } else {
            // ASCII format
            if (!get_line_safe(file, ascii_line)) {
                std::cerr << "Warning: Read failed or reached unexpected EOF at line " << i << "\n";
                break;
            }
            tokenize_to_views(ascii_line, tokens);
            if (tokens.size() < header.vertex_properties.size()) {
                std::cerr << "Warning: Incomplete ASCII vertex data at index " << i << ". Skipping.\n";
                continue;
            }
            
            x = parse_ascii_view(tokens[header.idx_x], header.vertex_properties[header.idx_x].type);
            y = parse_ascii_view(tokens[header.idx_y], header.vertex_properties[header.idx_y].type);
            z = parse_ascii_view(tokens[header.idx_z], header.vertex_properties[header.idx_z].type);
        }
        
        // Update bounding box
        orig_min_x = std::min(orig_min_x, x);
        orig_min_y = std::min(orig_min_y, y);
        orig_min_z = std::min(orig_min_z, z);
        orig_max_x = std::max(orig_max_x, x);
        orig_max_y = std::max(orig_max_y, y);
        orig_max_z = std::max(orig_max_z, z);
        
        // Voxel check
        int64_t gx = static_cast<int64_t>(std::floor(x / min_distance));
        int64_t gy = static_cast<int64_t>(std::floor(y / min_distance));
        int64_t gz = static_cast<int64_t>(std::floor(z / min_distance));
        VoxelKey key = { gx, gy, gz };
        
        if (voxel_grid.insert(key).second) {
            // New cell occupied! Decode remaining parameters and insert to Point_set_3
            Point p(x, y, z);
            Point_set::Index idx = *(point_set.insert(p));
            
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
                // ASCII decode properties
                if (has_normals) {
                    nx = parse_ascii_view(tokens[header.idx_nx], header.vertex_properties[header.idx_nx].type);
                    ny = parse_ascii_view(tokens[header.idx_ny], header.vertex_properties[header.idx_ny].type);
                    nz = parse_ascii_view(tokens[header.idx_nz], header.vertex_properties[header.idx_nz].type);
                }
                if (has_colors) {
                    r = parse_ascii_view(tokens[header.idx_red], header.vertex_properties[header.idx_red].type);
                    g = parse_ascii_view(tokens[header.idx_green], header.vertex_properties[header.idx_green].type);
                    b = parse_ascii_view(tokens[header.idx_blue], header.vertex_properties[header.idx_blue].type);
                }
                if (has_intensity) {
                    intensity = parse_ascii_view(tokens[header.idx_intensity], header.vertex_properties[header.idx_intensity].type);
                }
            }
            
            if (has_normals) {
                point_set.normal(idx) = Vector(nx, ny, nz);
            }
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
        
        // Progress logger
        if ((i + 1) % 10000000 == 0 || i + 1 == header.vertex_count) {
            double progress_pct = 100.0 * (i + 1) / header.vertex_count;
            double est_mem_mb = (voxel_grid.size() * 56.0 + point_set.size() * 56.0) / (1024.0 * 1024.0);
            std::cout << "Parsed: " << (i + 1) << " / " << header.vertex_count 
                      << " (" << std::fixed << std::setprecision(1) << progress_pct << "%), "
                      << "Decimated: " << point_set.size() << " points, "
                      << "Est. RAM: " << std::setprecision(1) << est_mem_mb << " MB\n";
        }
    }
    
    file.close();
    
    std::size_t decimated_count = point_set.size();
    double reduction_pct = 100.0 * (1.0 - (double)decimated_count / header.vertex_count);
    
    std::cout << "--------------------------------------------------\n";
    std::cout << "Streaming Phase Completed:\n";
    std::cout << "  Decimated Point Count: " << decimated_count << " / " << header.vertex_count << "\n";
    std::cout << "  Reduction Percentage:  " << std::fixed << std::setprecision(2) << reduction_pct << "%\n";
    std::cout << "--------------------------------------------------\n";

    // Garbage collection to release memory before heavy tasks
    point_set.collect_garbage();

    // WLOP Refinement Phase Trigger
    bool run_wlop = (reduction_pct >= 90.0);
    
    if (run_wlop) {
        std::cout << "Point count reduced by >= 90% (" << reduction_pct << "%). Executing WLOP refinement phase...\n";
        
        std::vector<Point> wlop_output;
        wlop_output.reserve(decimated_count);
        
        double wlop_radius = 2.0 * min_distance;
        
        auto wlop_start = std::chrono::high_resolution_clock::now();
        
        CGAL::wlop_simplify_and_regularize_point_set<CGAL::Parallel_tag>(
            point_set,
            std::back_inserter(wlop_output),
            CGAL::parameters::point_map(point_set.point_map())
                             .select_percentage(wlop_retain_percent)
                             .neighbor_radius(wlop_radius)
                             .number_of_iterations(wlop_iterations)
                             .require_uniform_sampling(wlop_require_uniform)
        );
        
        auto wlop_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> wlop_duration = wlop_end - wlop_start;
        
        std::cout << "WLOP executed in: " << wlop_duration.count() << " seconds. Output count: " << wlop_output.size() << ".\n";
        
        if (wlop_output.empty()) {
            std::cerr << "Error: WLOP returned 0 points. Skipping refinement.\n";
        } else {
            std::cout << "Initializing 1-NN attributes mapping KD-tree on decimated points...\n";
            
            // Build Search Tree on the pre-WLOP decimated set
            Point_set_property_map ppmap(&point_set);
            Traits traits(ppmap);
            Distance distance(ppmap);
            Search_tree search_tree(point_set.begin(), point_set.end(), Search_tree::Splitter(), traits);
            
            std::cout << "KD-tree built. Transferring attributes to regularized points...\n";
            
            // Setup final Point Set
            Point_set final_point_set;
            if (has_normals) final_point_set.add_normal_map();
            
            Point_set::Property_map<unsigned char> f_red_map, f_green_map, f_blue_map;
            if (has_colors) {
                f_red_map = final_point_set.add_property_map<unsigned char>("red", 0).first;
                f_green_map = final_point_set.add_property_map<unsigned char>("green", 0).first;
                f_blue_map = final_point_set.add_property_map<unsigned char>("blue", 0).first;
            }
            
            Point_set::Property_map<float> f_intensity_map_float;
            Point_set::Property_map<double> f_intensity_map_double;
            if (has_intensity) {
                if (intensity_is_double) {
                    f_intensity_map_double = final_point_set.add_property_map<double>(intensity_prop_name, 0.0).first;
                } else {
                    f_intensity_map_float = final_point_set.add_property_map<float>(intensity_prop_name, 0.0f).first;
                }
            }
            
            // Perform 1-NN query for each WLOP point
            for (const auto& w_pt : wlop_output) {
                Neighbor_search search(search_tree, w_pt, 1, 0.0, true, distance);
                if (search.begin() != search.end()) {
                    auto old_idx = search.begin()->first;
                    Point_set::Index new_idx = *(final_point_set.insert(w_pt));
                    
                    if (has_normals) {
                        final_point_set.normal(new_idx) = point_set.normal(old_idx);
                    }
                    if (has_colors) {
                        f_red_map[new_idx] = red_map[old_idx];
                        f_green_map[new_idx] = green_map[old_idx];
                        f_blue_map[new_idx] = blue_map[old_idx];
                    }
                    if (has_intensity) {
                        if (intensity_is_double) {
                            f_intensity_map_double[new_idx] = intensity_map_double[old_idx];
                        } else {
                            f_intensity_map_float[new_idx] = intensity_map_float[old_idx];
                        }
                    }
                }
            }
            
            // Swap to finalize
            point_set = std::move(final_point_set);
            
            // Clear maps to free up RAM immediately
            voxel_grid.clear();
            final_point_set.clear();
            final_point_set.collect_garbage();
            std::cout << "Attributes transfer complete.\n";
        }
    } else {
        std::cout << "Warning: Point count reduction is less than 90% (" << reduction_pct << "%). Skipping WLOP refinement phase to preserve RAM and prevent excessive processing times.\n";
        voxel_grid.clear();
    }
    
    // Compute decimated bounding box for verification
    double dec_min_x = std::numeric_limits<double>::infinity();
    double dec_min_y = std::numeric_limits<double>::infinity();
    double dec_min_z = std::numeric_limits<double>::infinity();
    double dec_max_x = -std::numeric_limits<double>::infinity();
    double dec_max_y = -std::numeric_limits<double>::infinity();
    double dec_max_z = -std::numeric_limits<double>::infinity();
    
    for (const auto& idx : point_set) {
        Point p = point_set.point(idx);
        dec_min_x = std::min(dec_min_x, p.x());
        dec_min_y = std::min(dec_min_y, p.y());
        dec_min_z = std::min(dec_min_z, p.z());
        dec_max_x = std::max(dec_max_x, p.x());
        dec_max_y = std::max(dec_max_y, p.y());
        dec_max_z = std::max(dec_max_z, p.z());
    }
    
    // Check bounding box match
    bool bbox_ok = true;
    auto check_bound = [&](double orig, double dec, const std::string& name) {
        double diff = std::abs(orig - dec);
        if (diff > min_distance) {
            std::cerr << "Warning: Bounding box mismatch on " << name 
                      << " (Original: " << orig << ", Decimated: " << dec 
                      << ", Diff: " << diff << ", Tolerance: " << min_distance << ")\n";
            bbox_ok = false;
        } else {
            std::cout << "Bounding box check passed for " << name 
                      << " (Original: " << orig << ", Decimated: " << dec 
                      << ", Diff: " << diff << " <= " << min_distance << ")\n";
        }
    };
    
    std::cout << "--------------------------------------------------\n";
    std::cout << "Verifying Data Integrity (Bounding Box Check):\n";
    check_bound(orig_min_x, dec_min_x, "min_x");
    check_bound(orig_min_y, dec_min_y, "min_y");
    check_bound(orig_min_z, dec_min_z, "min_z");
    check_bound(orig_max_x, dec_max_x, "max_x");
    check_bound(orig_max_y, dec_max_y, "max_y");
    check_bound(orig_max_z, dec_max_z, "max_z");
    
    if (!bbox_ok) {
        std::cerr << "Warning: Decimated bounding box does not match original bounding box within min_distance tolerance.\n";
    } else {
        std::cout << "Data integrity verification successful!\n";
    }
    std::cout << "--------------------------------------------------\n";
    
    // Serialize output to binary PLY
    std::cout << "Serializing decimated point cloud to: " << output_path << "\n";
    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: cannot open output file " << output_path << "\n";
        return EXIT_FAILURE;
    }
    CGAL::IO::set_binary_mode(ofs);
    ofs << point_set;
    if (!ofs) {
        std::cerr << "Error: failed to write point set to " << output_path << "\n";
        return EXIT_FAILURE;
    }
    ofs.close();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;
    
    std::cout << "Spatial subsampling completed successfully!\n";
    std::cout << "  Output point count: " << point_set.size() << "\n";
    std::cout << "  Total execution time: " << std::fixed << std::setprecision(2) << total_duration.count() << " seconds.\n";
    std::cout << "--------------------------------------------------\n";

    return EXIT_SUCCESS;
}
