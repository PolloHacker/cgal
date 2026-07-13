#include "ply_io.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <unordered_set>
#include <cstring>
#include <algorithm>
#include <limits>
#include <iomanip>

#include <CGAL/Point_set_3/IO.h>

namespace mesh_reconstruction {

namespace {

enum class PlyType {
    INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64, UNKNOWN
};

/**
 * \brief Maps a PLY type string to the PlyType enum.
 */
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

/**
 * \brief Returns the byte size of a PlyType.
 */
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

/**
 * \brief Safe line-reading to handle both Windows (\r\n) and Linux (\n) newlines.
 */
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

/**
 * \brief Reads a numerical value from a binary buffer according to the property type.
 */
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

/**
 * \brief Fast tokenizer for splitting ASCII lines.
 */
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

/**
 * \brief Parses an ASCII token into a double based on the property type.
 */
inline double parse_ascii_value(const std::string& token, PlyType type) {
    if (type == PlyType::FLOAT32 || type == PlyType::FLOAT64) {
        return std::stod(token);
    } else {
        return static_cast<double>(std::stoll(token));
    }
}

/**
 * \brief Converts a raw numerical color value into an unsigned char [0, 255].
 */
inline unsigned char to_uchar(double val, PlyType type) {
    if (type == PlyType::FLOAT32 || type == PlyType::FLOAT64) {
        if (val >= 0.0 && val <= 1.0) {
            return static_cast<unsigned char>(std::round(val * 255.0));
        }
    }
    return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
}

/**
 * \brief Parses a PLY file header, establishing property offsets and types.
 */
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

/**
 * \brief Per-voxel streaming decimation loader implementation.
 */
bool load_spatial_subsampled_ply(const std::string &filepath,
                                 Point_set &points,
                                 double min_distance) {
    PlyHeader header;
    std::ifstream file;
    if (!parse_ply_header(filepath, header, file)) {
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

} // namespace

bool load_ply(const std::string &filepath,
              Point_set &points,
              bool enable_spatial_subsampling,
              double min_distance) {
    if (enable_spatial_subsampling) {
        return load_spatial_subsampled_ply(filepath, points, min_distance);
    }
    
    // Normal loader delegates to CGAL native IO
    if (!CGAL::IO::read_point_set(filepath, points)) {
        std::cerr << "Error: cannot read point set from " << filepath << "\n";
        return false;
    }
    
    // Fallback zero normal generation if normals are completely absent
    if (!points.has_normal_map()) {
        points.add_normal_map();
        for (const auto &idx : points) {
            points.normal(idx) = Vector(0.0, 0.0, 0.0);
        }
        std::cout << "Input normals were not found/readable. Falling back to zero normals.\n";
    }
    
    return true;
}

bool write_ply(const std::string &filepath,
               const Point_set &points,
               bool binary) {
    if (!CGAL::IO::write_point_set(filepath, const_cast<Point_set&>(points), CGAL::parameters::use_binary_mode(binary))) {
        std::cerr << "Error: failed to write point set to " << filepath << "\n";
        return false;
    }
    return true;
}

} // namespace mesh_reconstruction
