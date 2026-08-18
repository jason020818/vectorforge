#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace vectorforge {

// Placeholder for later memory-layout work (alignment, pooling). Phase 0/1
// uses std::vector; do not add pooling until HNSW exists and is measured.
inline std::size_t aligned_float_count(std::size_t n, std::size_t alignment_bytes = 64) {
    if (alignment_bytes == 0 || alignment_bytes % sizeof(float) != 0) {
        throw std::invalid_argument("alignment must be a positive multiple of sizeof(float)");
    }
    const std::size_t stride = alignment_bytes / sizeof(float);
    return ((n + stride - 1) / stride) * stride;
}

}  // namespace vectorforge
