#pragma once

#include "vectorforge/types.hpp"

#include <cstddef>
#include <string>

namespace vectorforge {

class Index {
public:
    virtual ~Index() = default;

    [[nodiscard]] virtual dim_t dim() const = 0;
    [[nodiscard]] virtual Metric metric() const = 0;
    [[nodiscard]] virtual std::size_t size() const = 0;

    virtual void add(const float* vectors, std::size_t n) = 0;

    [[nodiscard]] virtual SearchResult search(const float* query, int k) const = 0;

    virtual void search_batch(const float* queries,
                              std::size_t nq,
                              int k,
                              idx_t* ids_out,
                              float* distances_out) const = 0;

    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
};

}  // namespace vectorforge
