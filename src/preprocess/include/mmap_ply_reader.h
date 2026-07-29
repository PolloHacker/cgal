#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>

namespace mesh_reconstruction {

/**
 * \brief RAII memory-mapped file reader for binary PLY files (Linux/POSIX).
 */
class MmapReader {
private:
    const char* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t header_offset_ = 0;
    int fd_ = -1;

public:
    MmapReader() = default;
    MmapReader(const std::string& filepath, std::size_t header_offset);
    ~MmapReader();

    MmapReader(const MmapReader&) = delete;
    MmapReader& operator=(const MmapReader&) = delete;

    MmapReader(MmapReader&& other) noexcept;
    MmapReader& operator=(MmapReader&& other) noexcept;

    void open(const std::string& filepath, std::size_t header_offset);
    void close() noexcept;

    [[nodiscard]] bool is_open() const noexcept { return data_ != nullptr; }
    [[nodiscard]] const char* data() const noexcept { return data_; }
    [[nodiscard]] const char* record_data() const noexcept { return data_ ? data_ + header_offset_ : nullptr; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t record_bytes() const noexcept { return size_ > header_offset_ ? size_ - header_offset_ : 0; }
    [[nodiscard]] std::size_t header_offset() const noexcept { return header_offset_; }
};

} // namespace mesh_reconstruction
