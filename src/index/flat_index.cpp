#include "vectorforge/flat_index.hpp"

#include "vectorforge/distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
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

std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(std::string(what) + " overflow");
    }
    return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* what) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error(std::string(what) + " overflow");
    }
    return a + b;
}

void validate_finite_block(const float* values, std::size_t count, const char* what) {
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            throw std::invalid_argument(std::string(what) + " contains non-finite values");
        }
    }
}

std::size_t current_stream_pos(std::ifstream& in) {
    const std::ifstream::pos_type pos = in.tellg();
    if (pos == std::ifstream::pos_type(-1)) {
        throw std::runtime_error("failed to inspect index file position");
    }
    return static_cast<std::size_t>(pos);
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
    const std::size_t dim = static_cast<std::size_t>(dim_);
    const std::size_t incoming = checked_mul(n, dim, "vector count * dim");
    const std::size_t old = vectors_.size();
    const std::size_t total = checked_add(old, incoming, "existing vectors + new vectors");
    validate_finite_block(vectors, incoming, "input vectors");
    vectors_.resize(total);
    std::copy(vectors, vectors + incoming, vectors_.begin() + static_cast<std::ptrdiff_t>(old));
    n_ = checked_add(n_, n, "index size");
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
    validate_finite_block(query, static_cast<std::size_t>(dim_), "query");

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
    const std::size_t total_queries =
        checked_mul(nq, static_cast<std::size_t>(dim_), "query count * dim");
    validate_finite_block(queries, total_queries, "queries");
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
        const std::size_t bytes =
            checked_mul(vectors_.size(), sizeof(float), "serialized vector bytes");
        out.write(reinterpret_cast<const char*>(vectors_.data()),
                  static_cast<std::streamsize>(bytes));
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

    const std::uint64_t max_size_t =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (n > max_size_t) {
        throw std::runtime_error("invalid vector count in index file");
    }

    const std::size_t file_size = static_cast<std::size_t>(std::filesystem::file_size(path));
    const std::size_t payload_offset = current_stream_pos(in);
    if (payload_offset > file_size) {
        throw std::runtime_error("invalid VectorForge index file: header past EOF");
    }

    const std::size_t dim_size = static_cast<std::size_t>(dim);
    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t element_count = checked_mul(n_size, dim_size, "serialized vector count");
    const std::size_t payload_bytes =
        checked_mul(element_count, sizeof(float), "serialized vector bytes");
    const std::size_t remaining_bytes = file_size - payload_offset;
    if (remaining_bytes < payload_bytes) {
        throw std::runtime_error("invalid VectorForge index file: truncated payload");
    }
    if (remaining_bytes > payload_bytes) {
        throw std::runtime_error("invalid VectorForge index file: unexpected trailing data");
    }

    std::vector<float> loaded_vectors(element_count, 0.0f);
    if (n_size > 0) {
        in.read(reinterpret_cast<char*>(loaded_vectors.data()),
                static_cast<std::streamsize>(payload_bytes));
        if (!in) {
            throw std::runtime_error("invalid VectorForge index file: truncated payload");
        }
    }

    validate_finite_block(loaded_vectors.data(), element_count, "serialized vectors");

    dim_ = static_cast<dim_t>(dim);
    metric_ = metric == 0u ? Metric::L2 : Metric::Cosine;
    n_ = n_size;
    vectors_ = std::move(loaded_vectors);
}

}  // namespace vectorforge
