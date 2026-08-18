#pragma once

#include "vectorforge/index.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace vectorforge {

// Exact (brute-force) search. This is the ground-truth engine used to score ANN
// recall. It is intentionally a scalar FP32 implementation.
class FlatIndex final : public Index {
public:
    FlatIndex(dim_t dim, Metric metric);
    FlatIndex(dim_t dim, const std::string& metric);

    [[nodiscard]] dim_t dim() const override { return dim_; }
    [[nodiscard]] Metric metric() const override { return metric_; }
    [[nodiscard]] std::size_t size() const override { return n_; }

    void add(const float* vectors, std::size_t n) override;

    [[nodiscard]] SearchResult search(const float* query, int k) const override;

    void search_batch(const float* queries,
                      std::size_t nq,
                      int k,
                      idx_t* ids_out,
                      float* distances_out) const override;

    void save(const std::string& path) const override;
    void load(const std::string& path) override;

    [[nodiscard]] const float* data() const { return vectors_.data(); }

private:
    void search_one(const float* query, int k, idx_t* ids_out, float* distances_out) const;

    dim_t dim_;
    Metric metric_;
    std::size_t n_{0};
    std::vector<float> vectors_;
};

}  // namespace vectorforge
