#include "vectorforge/flat_index.hpp"

#include "vectorforge/distance.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace vectorforge {
namespace {

constexpr float kPadDistance = std::numeric_limits<float>::infinity();
constexpr idx_t kPadId = -1;

struct Candidate {
    float distance;
    idx_t id;
};

// Max-heap of the current top-k: the worst candidate is on top and can be replaced.
// "Better" means smaller distance, then smaller id (deterministic ties).
struct Better {
    bool operator()(const Candidate& a, const Candidate& b) const {
        if (a.distance != b.distance) {
            return a.distance < b.distance;
        }
        return a.id < b.id;
    }
};

void require_positive_dim(dim_t dim) {
    if (dim <= 0) {
        throw std::invalid_argument("dim must be positive");
    }
}

}  // namespace

FlatIndex::FlatIndex(dim_t dim, Metric metric) : dim_(dim), metric_(metric), n_(0) {
    require_positive_dim(dim_);
}

FlatIndex::FlatIndex(dim_t dim, const std::string& metric) : FlatIndex(dim, parse_metric(metric)) {}

void FlatIndex::add(const float* vectors, std::size_t n) {
    if (n == 0) {
        return;
    }
    if (vectors == nullptr) {
        throw std::invalid_argument("vectors pointer is null");
    }
    const std::size_t old = vectors_.size();
    vectors_.resize(old + n * static_cast<std::size_t>(dim_));
    std::copy(vectors,
              vectors + n * static_cast<std::size_t>(dim_),
              vectors_.begin() + static_cast<std::ptrdiff_t>(old));
    n_ += n;
}

void FlatIndex::search_one(const float* query, int k, idx_t* ids_out, float* distances_out) const {
    if (query == nullptr) {
        throw std::invalid_argument("query pointer is null");
    }
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (ids_out == nullptr || distances_out == nullptr) {
        throw std::invalid_argument("output pointers are null");
    }

    const int k_out = k;
    for (int i = 0; i < k_out; ++i) {
        ids_out[i] = kPadId;
        distances_out[i] = kPadDistance;
    }
    if (n_ == 0) {
        return;
    }

    std::priority_queue<Candidate, std::vector<Candidate>, Better> heap;
    for (std::size_t i = 0; i < n_; ++i) {
        const float* vec = vectors_.data() + i * static_cast<std::size_t>(dim_);
        const Candidate cand{distance::compute(metric_, query, vec, dim_), static_cast<idx_t>(i)};
        if (heap.size() < static_cast<std::size_t>(k)) {
            heap.push(cand);
        } else if (Better{}(cand, heap.top())) {
            heap.pop();
            heap.push(cand);
        }
    }

    const int filled = static_cast<int>(heap.size());
    for (int pos = filled - 1; pos >= 0; --pos) {
        ids_out[pos] = heap.top().id;
        distances_out[pos] = heap.top().distance;
        heap.pop();
    }
}

SearchResult FlatIndex::search(const float* query, int k) const {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    SearchResult result;
    result.ids.assign(static_cast<std::size_t>(k), kPadId);
    result.distances.assign(static_cast<std::size_t>(k), kPadDistance);
    search_one(query, k, result.ids.data(), result.distances.data());
    return result;
}

void FlatIndex::search_batch(
    const float* queries, std::size_t nq, int k, idx_t* ids_out, float* distances_out) const {
    if (nq == 0) {
        return;
    }
    if (queries == nullptr) {
        throw std::invalid_argument("queries pointer is null");
    }
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (ids_out == nullptr || distances_out == nullptr) {
        throw std::invalid_argument("output pointers are null");
    }
    for (std::size_t q = 0; q < nq; ++q) {
        const float* query = queries + q * static_cast<std::size_t>(dim_);
        search_one(query,
                   k,
                   ids_out + q * static_cast<std::size_t>(k),
                   distances_out + q * static_cast<std::size_t>(k));
    }
}

namespace {

constexpr char kMagic[4] = {'V', 'F', '0', '1'};
constexpr std::uint32_t kKindFlat = 1;

}  // namespace

void FlatIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open '" + path + "' for writing");
    }
    out.write(kMagic, 4);
    const std::uint32_t kind = kKindFlat;
    const std::uint32_t dim = static_cast<std::uint32_t>(dim_);
    const std::uint32_t metric = metric_ == Metric::L2 ? 0u : 1u;
    const std::uint64_t n = n_;
    out.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (n_ > 0) {
        out.write(reinterpret_cast<const char*>(vectors_.data()),
                  static_cast<std::streamsize>(vectors_.size() * sizeof(float)));
    }
    if (!out) {
        throw std::runtime_error("failed while writing '" + path + "'");
    }
}

void FlatIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open '" + path + "' for reading");
    }
    char magic[4] = {};
    in.read(magic, 4);
    if (!in || magic[0] != kMagic[0] || magic[1] != kMagic[1] || magic[2] != kMagic[2] ||
        magic[3] != kMagic[3]) {
        throw std::runtime_error("invalid VectorForge index file: bad magic");
    }
    std::uint32_t kind = 0;
    std::uint32_t dim = 0;
    std::uint32_t metric = 0;
    std::uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&kind), sizeof(kind));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&metric), sizeof(metric));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in) {
        throw std::runtime_error("invalid VectorForge index file: truncated header");
    }
    if (kind != kKindFlat) {
        throw std::runtime_error("index kind is not FlatIndex");
    }
    if (dim == 0 || dim > static_cast<std::uint32_t>(std::numeric_limits<dim_t>::max())) {
        throw std::runtime_error("invalid dim in index file");
    }
    if (metric > 1u) {
        throw std::runtime_error("invalid metric in index file");
    }
    dim_ = static_cast<dim_t>(dim);
    metric_ = metric == 0u ? Metric::L2 : Metric::Cosine;
    n_ = static_cast<std::size_t>(n);
    vectors_.assign(n_ * static_cast<std::size_t>(dim_), 0.0f);
    if (n_ > 0) {
        in.read(reinterpret_cast<char*>(vectors_.data()),
                static_cast<std::streamsize>(vectors_.size() * sizeof(float)));
        if (!in) {
            throw std::runtime_error("invalid VectorForge index file: truncated vectors");
        }
    }
}

}  // namespace vectorforge
