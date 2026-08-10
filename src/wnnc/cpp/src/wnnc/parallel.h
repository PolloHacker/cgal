// SPDX-License-Identifier: MIT
// Part of the C++ port of WNNC (Lin et al., ACM ToG 2024).
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace wnnc {

/// Runs `body(index)` for every index in [0, count), across hardware threads.
/// Threads grab fixed-size chunks from a shared counter, so uneven per-index
/// workloads (treecode traversals, heterogeneous performance/efficiency
/// cores) stay balanced. Blocks until all indices are processed.
/// `body` must be safe to call concurrently for distinct indices.
template <typename IndexBody>
void parallelFor(std::int64_t count, const IndexBody& body, std::int64_t chunkSize = 128) {
    if (count <= 0) {
        return;
    }

    const auto hardwareThreads = static_cast<std::int64_t>(std::thread::hardware_concurrency());
    const std::int64_t chunkCount = (count + chunkSize - 1) / chunkSize;
    const std::int64_t threadCount = std::clamp<std::int64_t>(hardwareThreads, 1, chunkCount);
    if (threadCount == 1) {
        for (std::int64_t i = 0; i < count; ++i) {
            body(i);
        }
        return;
    }

    std::atomic<std::int64_t> nextChunkBegin{0};
    const auto worker = [&count, &body, &nextChunkBegin, chunkSize] {
        while (true) {
            const std::int64_t begin =
                nextChunkBegin.fetch_add(chunkSize, std::memory_order_relaxed);
            if (begin >= count) {
                return;
            }
            const std::int64_t end = std::min(begin + chunkSize, count);
            for (std::int64_t i = begin; i < end; ++i) {
                body(i);
            }
        }
    };

    std::vector<std::thread> helpers;
    helpers.reserve(static_cast<std::size_t>(threadCount - 1));
    for (std::int64_t t = 1; t < threadCount; ++t) {
        helpers.emplace_back(worker);
    }
    worker();  // the calling thread participates as well
    for (std::thread& helper : helpers) {
        helper.join();
    }
}

}  // namespace wnnc
