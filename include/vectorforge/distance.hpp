#pragma once

#include "vectorforge/types.hpp"

namespace vectorforge::distance {

// Scalar FP32 reference kernels. SIMD variants are deferred until after HNSW
// (see ARCHITECTURE.md). Results of these kernels are the correctness baseline.

float l2(const float* a, const float* b, dim_t dim);

float cosine_similarity(const float* a, const float* b, dim_t dim);

// Cosine distance = 1 - cosine_similarity. Smaller is closer.
// Zero-norm vectors are defined to have similarity 0 with every vector,
// including other zero-norm vectors, so distance is 1.
float cosine_distance(const float* a, const float* b, dim_t dim);

inline float compute(Metric metric, const float* a, const float* b, dim_t dim) {
    switch (metric) {
        case Metric::L2:
            return l2(a, b, dim);
        case Metric::Cosine:
            return cosine_distance(a, b, dim);
    }
    throw std::invalid_argument("invalid metric enum");
}

}  // namespace vectorforge::distance
