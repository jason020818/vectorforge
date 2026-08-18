#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace vectorforge {

using idx_t = std::int64_t;
using dim_t = int;

enum class Metric { L2, Cosine };

inline Metric parse_metric(const std::string& name) {
    if (name == "l2" || name == "L2" || name == "euclidean") {
        return Metric::L2;
    }
    if (name == "cosine" || name == "cos" || name == "Cosine") {
        return Metric::Cosine;
    }
    throw std::invalid_argument("unknown metric '" + name + "'; expected 'l2' or 'cosine'");
}

inline const char* metric_name(Metric metric) {
    switch (metric) {
        case Metric::L2:
            return "l2";
        case Metric::Cosine:
            return "cosine";
    }
    throw std::invalid_argument("invalid metric enum");
}

struct SearchResult {
    std::vector<idx_t> ids;
    std::vector<float> distances;
};

}  // namespace vectorforge
