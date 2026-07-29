#pragma once

#include <cstddef>
#include <stdexcept>
#include <span>
#include <utility>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace mesh_reconstruction {

/**
 * \brief RAII wrapper for CUDA pinned (page-locked) host memory.
 */
template <typename T>
class PinnedBuffer {
private:
    T* data_ = nullptr;
    std::size_t capacity_ = 0;

public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t capacity) {
        allocate(capacity);
    }

    ~PinnedBuffer() {
        free();
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept
        : data_(other.data_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.capacity_ = 0;
    }

    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.capacity_ = 0;
        }
        return *this;
    }

    void allocate(std::size_t capacity) {
        free();
        if (capacity == 0) return;

#ifdef USE_CUDA
        cudaError_t err = cudaHostAlloc(reinterpret_cast<void**>(&data_), capacity * sizeof(T), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            data_ = nullptr;
            capacity_ = 0;
            throw std::runtime_error("cudaHostAlloc failed");
        }
#else
        data_ = new T[capacity];
#endif
        capacity_ = capacity;
    }

    void free() noexcept {
        if (data_) {
#ifdef USE_CUDA
            cudaFreeHost(data_);
#else
            delete[] data_;
#endif
            data_ = nullptr;
            capacity_ = 0;
        }
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::span<T> span(std::size_t count) noexcept {
        return std::span<T>(data_, std::min(count, capacity_));
    }

    [[nodiscard]] std::span<const T> span(std::size_t count) const noexcept {
        return std::span<const T>(data_, std::min(count, capacity_));
    }
};

} // namespace mesh_reconstruction
