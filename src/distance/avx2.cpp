// AVX2 distance kernels are deferred until after the HNSW baseline (Phase SIMD).
// This translation unit exists so the planned layout stays in the tree without
// shipping an unused / untested kernel. Do not call from production code yet.

#include "vectorforge/distance.hpp"

namespace vectorforge::distance::avx2 {

// Intentionally empty: feature detection + kernels land in v0.2.

}  // namespace vectorforge::distance::avx2
