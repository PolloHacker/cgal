// SPDX-License-Identifier: MIT
// Part of the C++ port of WNNC (Lin et al., ACM ToG 2024).
#include "point_cloud_io.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace wnnc::io {
namespace {

static_assert(std::endian::native == std::endian::little,
              "binary .ply/.npy readers assume a little-endian host");

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& reason) {
    throw std::runtime_error(path.string() + ": " + reason);
}

std::ifstream openFile(const std::filesystem::path& path, std::ios::openmode mode) {
    std::ifstream file(path, mode);
    if (!file) {
        fail(path, "cannot open file");
    }
    return file;
}

/// Parses the first three whitespace-separated numbers of `line` into `point`.
/// Returns false for lines with fewer than three numbers (e.g. blank lines).
bool parsePointLine(const std::string& line, Vec3<double>& point) {
    const char* cursor = line.c_str();
    std::array<double, 3> values{};
    for (double& value : values) {
        char* parseEnd = nullptr;
        value = std::strtod(cursor, &parseEnd);
        if (parseEnd == cursor) {
            return false;
        }
        cursor = parseEnd;
    }
    point = {values[0], values[1], values[2]};
    return true;
}

std::vector<Vec3<double>> loadXyz(const std::filesystem::path& path) {
    std::ifstream file = openFile(path, std::ios::in);
    std::vector<Vec3<double>> points;
    std::string line;
    while (std::getline(file, line)) {
        if (Vec3<double> point; parsePointLine(line, point)) {
            points.push_back(point);
        }
    }
    return points;
}

std::vector<Vec3<double>> loadObj(const std::filesystem::path& path) {
    std::ifstream file = openFile(path, std::ios::in);
    std::vector<Vec3<double>> points;
    std::string line;
    while (std::getline(file, line)) {
        // Vertex lines look like "v x y z [w]"; everything else is ignored.
        if (line.size() < 2 || line[0] != 'v' || !std::isspace(static_cast<unsigned char>(line[1]))) {
            continue;
        }
        if (Vec3<double> point; parsePointLine(line.substr(1), point)) {
            points.push_back(point);
        }
    }
    return points;
}

//// .npy ////

std::vector<Vec3<double>> loadNpy(const std::filesystem::path& path) {
    std::ifstream file = openFile(path, std::ios::binary);

    std::array<char, 8> preamble{};  // magic string + version
    file.read(preamble.data(), preamble.size());
    if (!file || std::memcmp(preamble.data(), "\x93NUMPY", 6) != 0) {
        fail(path, "not a .npy file");
    }

    const int versionMajor = preamble[6];
    std::uint32_t headerLength = 0;
    if (versionMajor == 1) {
        std::uint16_t shortLength = 0;
        file.read(reinterpret_cast<char*>(&shortLength), sizeof(shortLength));
        headerLength = shortLength;
    } else {
        file.read(reinterpret_cast<char*>(&headerLength), sizeof(headerLength));
    }

    std::string header(headerLength, '\0');
    file.read(header.data(), headerLength);
    if (!file) {
        fail(path, "truncated .npy header");
    }

    // The header is a Python dict literal, e.g.
    // {'descr': '<f8', 'fortran_order': False, 'shape': (40000, 3), }
    const bool isFloat32 = header.find("'<f4'") != std::string::npos;
    const bool isFloat64 = header.find("'<f8'") != std::string::npos;
    if (!isFloat32 && !isFloat64) {
        fail(path, "unsupported .npy dtype (expected '<f4' or '<f8')");
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        fail(path, "unsupported .npy layout (expected C-contiguous)");
    }

    const std::size_t shapeStart = header.find("'shape': (");
    if (shapeStart == std::string::npos) {
        fail(path, "malformed .npy header");
    }
    std::size_t rowCount = 0;
    std::size_t columnCount = 0;
    if (std::sscanf(header.c_str() + shapeStart, "'shape': (%zu, %zu", &rowCount,
                    &columnCount) != 2 ||
        columnCount < 3) {
        fail(path, "expected a 2D array with at least 3 columns");
    }

    const std::size_t elementSize = isFloat64 ? sizeof(double) : sizeof(float);
    std::vector<char> row(columnCount * elementSize);
    std::vector<Vec3<double>> points;
    points.reserve(rowCount);
    for (std::size_t i = 0; i < rowCount; ++i) {
        file.read(row.data(), static_cast<std::streamsize>(row.size()));
        if (!file) {
            fail(path, "truncated .npy data");
        }
        Vec3<double> point;
        if (isFloat64) {
            std::array<double, 3> xyz{};
            std::memcpy(xyz.data(), row.data(), sizeof(xyz));
            point = {xyz[0], xyz[1], xyz[2]};
        } else {
            std::array<float, 3> xyz{};
            std::memcpy(xyz.data(), row.data(), sizeof(xyz));
            point = {xyz[0], xyz[1], xyz[2]};
        }
        points.push_back(point);
    }
    return points;
}

//// .ply ////

struct PlyProperty {
    std::string name;
    std::size_t byteSize = 0;
    bool isFloat32 = false;
    bool isFloat64 = false;
};

std::size_t plyTypeSize(const std::string& type) {
    if (type == "char" || type == "uchar" || type == "int8" || type == "uint8") return 1;
    if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") return 2;
    if (type == "int" || type == "uint" || type == "int32" || type == "uint32" ||
        type == "float" || type == "float32") return 4;
    if (type == "double" || type == "float64") return 8;
    return 0;
}

std::vector<Vec3<double>> loadPly(const std::filesystem::path& path) {
    std::ifstream file = openFile(path, std::ios::binary);

    bool isBinary = false;
    std::size_t vertexCount = 0;
    std::vector<PlyProperty> vertexProperties;
    bool inVertexElement = false;

    std::string line;
    if (!std::getline(file, line) || line.rfind("ply", 0) != 0) {
        fail(path, "not a .ply file");
    }
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;

        if (keyword == "format") {
            std::string format;
            tokens >> format;
            if (format == "binary_little_endian") {
                isBinary = true;
            } else if (format != "ascii") {
                fail(path, "unsupported .ply format: " + format);
            }
        } else if (keyword == "element") {
            std::string name;
            std::size_t count = 0;
            tokens >> name >> count;
            if (name == "vertex") {
                inVertexElement = true;
                vertexCount = count;
            } else {
                if (vertexCount == 0) {
                    fail(path, "only .ply files whose first element is 'vertex' are supported");
                }
                inVertexElement = false;  // faces etc. follow the vertices; ignore them
            }
        } else if (keyword == "property" && inVertexElement) {
            std::string type;
            std::string name;
            tokens >> type >> name;
            if (type == "list") {
                fail(path, "list properties in the vertex element are not supported");
            }
            const std::size_t byteSize = plyTypeSize(type);
            if (byteSize == 0) {
                fail(path, "unknown .ply property type: " + type);
            }
            vertexProperties.push_back(
                {name, byteSize, type == "float" || type == "float32",
                 type == "double" || type == "float64"});
        } else if (keyword == "end_header") {
            break;
        }
    }

    // Locate x/y/z within the vertex record.
    std::array<std::size_t, 3> coordOffset{};
    std::array<const PlyProperty*, 3> coordProperty{};
    std::size_t recordSize = 0;
    for (const PlyProperty& property : vertexProperties) {
        const int axis = property.name == "x" ? 0 : property.name == "y" ? 1
                       : property.name == "z" ? 2 : -1;
        if (axis >= 0) {
            coordOffset[static_cast<std::size_t>(axis)] = recordSize;
            coordProperty[static_cast<std::size_t>(axis)] = &property;
        }
        recordSize += property.byteSize;
    }
    for (const PlyProperty* property : coordProperty) {
        if (property == nullptr) {
            fail(path, "vertex element lacks x/y/z properties");
        }
        if (!property->isFloat32 && !property->isFloat64) {
            fail(path, "x/y/z properties must be float or double");
        }
    }

    std::vector<Vec3<double>> points;
    points.reserve(vertexCount);
    if (isBinary) {
        std::vector<char> record(recordSize);
        for (std::size_t i = 0; i < vertexCount; ++i) {
            file.read(record.data(), static_cast<std::streamsize>(record.size()));
            if (!file) {
                fail(path, "truncated .ply vertex data");
            }
            std::array<double, 3> xyz{};
            for (int axis = 0; axis < 3; ++axis) {
                const char* source = record.data() + coordOffset[static_cast<std::size_t>(axis)];
                if (coordProperty[static_cast<std::size_t>(axis)]->isFloat64) {
                    double value;
                    std::memcpy(&value, source, sizeof(value));
                    xyz[static_cast<std::size_t>(axis)] = value;
                } else {
                    float value;
                    std::memcpy(&value, source, sizeof(value));
                    xyz[static_cast<std::size_t>(axis)] = value;
                }
            }
            points.push_back({xyz[0], xyz[1], xyz[2]});
        }
    } else {
        // ASCII: x/y/z are token positions, independent of the byte layout.
        std::array<std::size_t, 3> coordColumn{};
        for (std::size_t column = 0; column < vertexProperties.size(); ++column) {
            const std::string& name = vertexProperties[column].name;
            if (name == "x") coordColumn[0] = column;
            if (name == "y") coordColumn[1] = column;
            if (name == "z") coordColumn[2] = column;
        }
        for (std::size_t i = 0; i < vertexCount; ++i) {
            if (!std::getline(file, line)) {
                fail(path, "truncated .ply vertex data");
            }
            std::istringstream tokens(line);
            std::vector<double> values(vertexProperties.size());
            for (double& value : values) {
                if (!(tokens >> value)) {
                    fail(path, "malformed .ply vertex line");
                }
            }
            points.push_back({values[coordColumn[0]], values[coordColumn[1]],
                              values[coordColumn[2]]});
        }
    }
    return points;
}

}  // namespace

std::vector<Vec3<double>> loadPoints(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    std::vector<Vec3<double>> points;
    if (extension == ".xyz") {
        points = loadXyz(path);
    } else if (extension == ".obj") {
        points = loadObj(path);
    } else if (extension == ".ply") {
        points = loadPly(path);
    } else if (extension == ".npy") {
        points = loadNpy(path);
    } else {
        fail(path, "unsupported extension (expected .xyz, .obj, .ply or .npy)");
    }

    if (points.empty()) {
        fail(path, "no points found");
    }
    return points;
}

void saveXyzWithNormals(const std::filesystem::path& path, std::span<const Vec3<double>> points,
                        std::span<const Vec3<double>> normals) {
    if (points.size() != normals.size()) {
        throw std::invalid_argument("saveXyzWithNormals: points/normals size mismatch");
    }
    std::ofstream file(path);
    if (!file) {
        fail(path, "cannot open file for writing");
    }
    std::array<char, 256> line{};
    for (std::size_t i = 0; i < points.size(); ++i) {
        const Vec3<double>& p = points[i];
        const Vec3<double>& n = normals[i];
        const int length = std::snprintf(line.data(), line.size(),
                                         "%.10g %.10g %.10g %.10g %.10g %.10g\n", p.x, p.y, p.z,
                                         n.x, n.y, n.z);
        file.write(line.data(), length);
    }
    if (!file) {
        fail(path, "write failed");
    }
}

}  // namespace wnnc::io
