#include "mmap_ply_reader.h"

#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mesh_reconstruction {

MmapReader::MmapReader(const std::string& filepath, std::size_t header_offset) {
    open(filepath, header_offset);
}

MmapReader::~MmapReader() {
    close();
}

MmapReader::MmapReader(MmapReader&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
      header_offset_(other.header_offset_),
      fd_(other.fd_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.header_offset_ = 0;
    other.fd_ = -1;
}

MmapReader& MmapReader::operator=(MmapReader&& other) noexcept {
    if (this != &other) {
        close();
        data_ = other.data_;
        size_ = other.size_;
        header_offset_ = other.header_offset_;
        fd_ = other.fd_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.header_offset_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

void MmapReader::open(const std::string& filepath, std::size_t header_offset) {
    close();
    header_offset_ = header_offset;

    int fd = ::open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("MmapReader: Failed to open file: " + filepath);
    }
    fd_ = fd;

    struct stat st;
    if (::fstat(fd, &st) < 0) {
        close();
        throw std::runtime_error("MmapReader: fstat failed: " + filepath);
    }
    size_ = static_cast<std::size_t>(st.st_size);

    if (size_ == 0) {
        return;
    }

    void* pMap = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (pMap == MAP_FAILED) {
        close();
        throw std::runtime_error("MmapReader: mmap failed: " + filepath);
    }
    data_ = static_cast<const char*>(pMap);
#ifdef MADV_SEQUENTIAL
    ::madvise(pMap, size_, MADV_SEQUENTIAL);
#endif
}

void MmapReader::close() noexcept {
    if (data_) {
        ::munmap(const_cast<char*>(data_), size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
    header_offset_ = 0;
}

} // namespace mesh_reconstruction
