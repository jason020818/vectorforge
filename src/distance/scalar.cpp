#include "vectorforge/distance.hpp"

#include <cmath>

namespace vectorforge::distance {

float l2(const float* a, const float* b, dim_t dim) {
    double acc = 0.0;
    for (dim_t i = 0; i < dim; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return static_cast<float>(std::sqrt(acc));
}

float cosine_similarity(const float* a, const float* b, dim_t dim) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (dim_t i = 0; i < dim; ++i) {
        const double av = static_cast<double>(a[i]);
        const double bv = static_cast<double>(b[i]);
        dot += av * bv;
        na += av * av;
        nb += bv * bv;
    }
    if (na == 0.0 || nb == 0.0) {
        return 0.0f;
    }
    return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

float cosine_distance(const float* a, const float* b, dim_t dim) {
    return 1.0f - cosine_similarity(a, b, dim);
}

}  // namespace vectorforge::distance
