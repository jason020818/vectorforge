#include "vectorforge/hnsw_index.hpp"

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
#include <string>
#include <utility>

namespace vectorforge {
namespace {

constexpr float kPadDistance = std::numeric_limits<float>::infinity();
constexpr idx_t kPadId = -1;
constexpr int kMaxLevelCap = 64;
constexpr char kMagic[4] = {'V', 'H', '0', '1'};

bool is_better_id(float da, idx_t ia, float db, idx_t ib) {
    if (da != db) {
        return da < db;
    }
    return ia < ib;
}

template <typename C>
bool is_better(const C& a, const C& b) {
    return is_better_id(a.distance, a.id, b.distance, b.id);
}

// Max-heap of W: furthest (worst) on top.
struct WorseOnTop {
    template <typename C>
    bool operator()(const C& a, const C& b) const {
        return is_better(a, b);
    }
};

// Min-heap of C: nearest (best) on top.
struct BetterOnTop {
    template <typename C>
    bool operator()(const C& a, const C& b) const {
        return is_better(b, a);
    }
};

void require_positive_dim(dim_t dim) {
    if (dim <= 0) {
        throw std::invalid_argument("dim must be positive");
    }
}

void require_hnsw_params(std::size_t M, std::size_t ef_construction, std::size_t ef_search) {
    if (M < 2) {
        throw std::invalid_argument("M must be at least 2");
    }
    if (M > std::numeric_limits<std::size_t>::max() / 2) {
        throw std::invalid_argument("M is too large");
    }
    if (ef_construction < M) {
        throw std::invalid_argument("efConstruction must be at least M");
    }
    if (ef_search == 0) {
        throw std::invalid_argument("efSearch must be positive");
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

void require_read(std::ifstream& in, char* dst, std::streamsize n, const char* what) {
    in.read(dst, n);
    if (!in || in.gcount() != n) {
        throw std::runtime_error(std::string("invalid VectorForge HNSW index file: ") + what);
    }
}

template <typename T>
void read_pod(std::ifstream& in, T& value, const char* what) {
    require_read(
        in, reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)), what);
}

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
}

std::uint64_t fnv1a64(std::uint64_t hash, std::uint64_t value) {
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (8 * i)) & 0xffULL;
        hash *= kPrime;
    }
    return hash;
}

std::uint64_t fnv1a64_i32(std::uint64_t hash, std::int32_t value) {
    return fnv1a64(hash, static_cast<std::uint64_t>(static_cast<std::uint32_t>(value)));
}

double next_open_unit(std::mt19937_64& rng) {
    // Portable (0, 1): 53-bit mantissa, offset by 0.5 ulp so U is never 0 or 1.
    const std::uint64_t bits = rng() >> 11;
    return (static_cast<double>(bits) + 0.5) * 0x1p-53;
}

}  // namespace

HNSWIndex::HNSWIndex(dim_t dim,
                     Metric metric,
                     std::size_t M,
                     std::size_t ef_construction,
                     std::size_t ef_search,
                     std::uint64_t seed)
    : dim_(dim),
      metric_(metric),
      n_(0),
      M_(M),
      M0_(0),
      ef_construction_(ef_construction),
      ef_search_(ef_search),
      seed_(seed),
      level_mult_(0.0),
      max_level_(-1),
      entry_point_(-1),
      rng_(seed) {
    require_positive_dim(dim_);
    require_hnsw_params(M_, ef_construction_, ef_search_);
    M0_ = 2 * M_;
    level_mult_ = 1.0 / std::log(static_cast<double>(M_));
}

HNSWIndex::HNSWIndex(dim_t dim,
                     const std::string& metric,
                     std::size_t M,
                     std::size_t ef_construction,
                     std::size_t ef_search,
                     std::uint64_t seed)
    : HNSWIndex(dim, parse_metric(metric), M, ef_construction, ef_search, seed) {}

void HNSWIndex::set_ef_search(std::size_t ef) {
    if (ef == 0) {
        throw std::invalid_argument("efSearch must be positive");
    }
    ef_search_ = ef;
}

int HNSWIndex::node_level(idx_t id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= n_) {
        throw std::invalid_argument("node id out of range");
    }
    return nodes_[static_cast<std::size_t>(id)].level;
}

std::vector<idx_t> HNSWIndex::neighbors(idx_t id, int layer) const {
    if (id < 0 || static_cast<std::size_t>(id) >= n_) {
        throw std::invalid_argument("node id out of range");
    }
    const Node& node = nodes_[static_cast<std::size_t>(id)];
    if (layer < 0 || layer > node.level) {
        throw std::invalid_argument("layer out of range for node");
    }
    return node.neighbors[static_cast<std::size_t>(layer)];
}

const float* HNSWIndex::vec(idx_t id) const {
    return vectors_.data() + static_cast<std::size_t>(id) * static_cast<std::size_t>(dim_);
}

float HNSWIndex::dist(const float* query, idx_t id) const {
    return distance::compute(metric_, query, vec(id), dim_);
}

int HNSWIndex::random_level() {
    const double u = next_open_unit(rng_);
    int level = static_cast<int>(std::floor(-std::log(u) * level_mult_));
    if (level < 0) {
        level = 0;
    }
    if (level > kMaxLevelCap) {
        level = kMaxLevelCap;
    }
    return level;
}

void HNSWIndex::restore_rng_after_n_inserts() {
    rng_.seed(seed_);
    for (std::size_t i = 0; i < n_; ++i) {
        (void)next_open_unit(rng_);
    }
}

std::vector<HNSWIndex::Candidate> HNSWIndex::search_layer(const float* query,
                                                          const std::vector<idx_t>& entries,
                                                          std::size_t ef,
                                                          int layer) const {
    std::vector<HNSWIndex::Candidate> empty;
    if (entries.empty() || n_ == 0 || ef == 0) {
        return empty;
    }

    if (visit_stamp_.size() < n_) {
        visit_stamp_.assign(n_, 0);
        visit_gen_ = 1;
    }
    std::uint32_t gen = visit_gen_++;
    if (gen == 0) {
        std::fill(visit_stamp_.begin(), visit_stamp_.end(), 0);
        visit_gen_ = 2;
        gen = 1;
    }

    std::priority_queue<HNSWIndex::Candidate, std::vector<HNSWIndex::Candidate>, BetterOnTop>
        candidates;
    std::priority_queue<HNSWIndex::Candidate, std::vector<HNSWIndex::Candidate>, WorseOnTop> w;

    for (idx_t ep : entries) {
        if (ep < 0 || static_cast<std::size_t>(ep) >= n_) {
            continue;
        }
        if (visit_stamp_[static_cast<std::size_t>(ep)] == gen) {
            continue;
        }
        visit_stamp_[static_cast<std::size_t>(ep)] = gen;
        const HNSWIndex::Candidate c{dist(query, ep), ep};
        candidates.push(c);
        w.push(c);
    }
    if (w.empty()) {
        return empty;
    }

    while (!candidates.empty()) {
        const HNSWIndex::Candidate c = candidates.top();
        candidates.pop();
        const HNSWIndex::Candidate furthest = w.top();
        if (c.distance > furthest.distance) {
            break;
        }
        const Node& node = nodes_[static_cast<std::size_t>(c.id)];
        if (layer > node.level) {
            continue;
        }
        const std::vector<idx_t>& neigh = node.neighbors[static_cast<std::size_t>(layer)];
        for (idx_t e : neigh) {
            if (e < 0 || static_cast<std::size_t>(e) >= n_) {
                continue;
            }
            if (visit_stamp_[static_cast<std::size_t>(e)] == gen) {
                continue;
            }
            visit_stamp_[static_cast<std::size_t>(e)] = gen;
            const HNSWIndex::Candidate ec{dist(query, e), e};
            const HNSWIndex::Candidate f = w.top();
            if (w.size() < ef || is_better(ec, f)) {
                candidates.push(ec);
                w.push(ec);
                if (w.size() > ef) {
                    w.pop();
                }
            }
        }
    }

    std::vector<HNSWIndex::Candidate> result;
    result.reserve(w.size());
    while (!w.empty()) {
        result.push_back(w.top());
        w.pop();
    }
    std::sort(result.begin(),
              result.end(),
              [](const HNSWIndex::Candidate& a, const HNSWIndex::Candidate& b) {
                  return is_better(a, b);
              });
    return result;
}

std::vector<idx_t> HNSWIndex::select_neighbors_heuristic(
    std::vector<HNSWIndex::Candidate> candidates, std::size_t m) const {
    std::sort(candidates.begin(),
              candidates.end(),
              [](const HNSWIndex::Candidate& a, const HNSWIndex::Candidate& b) {
                  return is_better(a, b);
              });

    std::vector<HNSWIndex::Candidate> selected;
    std::vector<HNSWIndex::Candidate> discarded;
    selected.reserve(m);
    for (const HNSWIndex::Candidate& e : candidates) {
        if (selected.size() >= m) {
            discarded.push_back(e);
            continue;
        }
        bool diverse = true;
        for (const HNSWIndex::Candidate& r : selected) {
            const float d_er = distance::compute(metric_, vec(e.id), vec(r.id), dim_);
            if (d_er < e.distance) {
                diverse = false;
                break;
            }
        }
        if (diverse) {
            selected.push_back(e);
        } else {
            discarded.push_back(e);
        }
    }
    // keepPrunedConnections = true: fill remaining slots from discarded nearest-first.
    for (const HNSWIndex::Candidate& e : discarded) {
        if (selected.size() >= m) {
            break;
        }
        selected.push_back(e);
    }

    std::vector<idx_t> ids;
    ids.reserve(selected.size());
    for (const HNSWIndex::Candidate& e : selected) {
        ids.push_back(e.id);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

void HNSWIndex::prune_neighbors(idx_t id, int layer, std::size_t max_degree) {
    Node& node = nodes_[static_cast<std::size_t>(id)];
    std::vector<idx_t>& neigh = node.neighbors[static_cast<std::size_t>(layer)];
    if (neigh.size() <= max_degree) {
        return;
    }
    const float* query = vec(id);
    std::vector<HNSWIndex::Candidate> candidates;
    candidates.reserve(neigh.size());
    for (idx_t nb : neigh) {
        candidates.push_back(HNSWIndex::Candidate{dist(query, nb), nb});
    }
    neigh = select_neighbors_heuristic(std::move(candidates), max_degree);
}

void HNSWIndex::insert_node(idx_t id) {
    const int level = random_level();
    Node node;
    node.level = level;
    node.neighbors.resize(static_cast<std::size_t>(level) + 1);
    nodes_[static_cast<std::size_t>(id)] = std::move(node);

    if (entry_point_ < 0) {
        entry_point_ = id;
        max_level_ = level;
        return;
    }

    const float* query = vec(id);
    std::vector<idx_t> enter{entry_point_};

    for (int lc = max_level_; lc > level; --lc) {
        const auto found = search_layer(query, enter, 1, lc);
        if (found.empty()) {
            break;
        }
        enter.assign(1, found.front().id);
    }

    const int connect_top = std::min(max_level_, level);
    for (int lc = connect_top; lc >= 0; --lc) {
        const auto found = search_layer(query, enter, ef_construction_, lc);
        if (found.empty()) {
            break;
        }
        enter.clear();
        enter.reserve(found.size());
        for (const HNSWIndex::Candidate& c : found) {
            enter.push_back(c.id);
        }

        const std::vector<idx_t> selected = select_neighbors_heuristic(found, M_);
        Node& cur = nodes_[static_cast<std::size_t>(id)];
        std::vector<idx_t>& cur_neigh = cur.neighbors[static_cast<std::size_t>(lc)];
        cur_neigh = selected;

        const std::size_t max_deg = (lc == 0) ? M0_ : M_;
        for (idx_t nb : selected) {
            Node& other = nodes_[static_cast<std::size_t>(nb)];
            if (lc > other.level) {
                continue;
            }
            std::vector<idx_t>& other_neigh = other.neighbors[static_cast<std::size_t>(lc)];
            if (std::find(other_neigh.begin(), other_neigh.end(), id) == other_neigh.end()) {
                other_neigh.push_back(id);
            }
            if (other_neigh.size() > max_deg) {
                prune_neighbors(nb, lc, max_deg);
            } else {
                std::sort(other_neigh.begin(), other_neigh.end());
            }
        }
    }

    if (level > max_level_) {
        entry_point_ = id;
        max_level_ = level;
    }
}

void HNSWIndex::add(const float* vectors, std::size_t n) {
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

    const std::size_t old_n = n_;
    const std::size_t new_n = checked_add(old_n, n, "index size");
    vectors_.resize(total);
    std::copy(vectors, vectors + incoming, vectors_.begin() + static_cast<std::ptrdiff_t>(old));
    nodes_.resize(new_n);
    n_ = new_n;
    visit_stamp_.assign(n_, 0);
    visit_gen_ = 1;

    for (std::size_t i = 0; i < n; ++i) {
        insert_node(static_cast<idx_t>(old_n + i));
    }
}

void HNSWIndex::search_one(const float* query, int k, idx_t* ids_out, float* distances_out) const {
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

    for (int i = 0; i < k; ++i) {
        ids_out[i] = kPadId;
        distances_out[i] = kPadDistance;
    }
    if (n_ == 0 || entry_point_ < 0) {
        return;
    }

    const std::size_t k_sz = static_cast<std::size_t>(k);
    const std::size_t effective_ef = std::max(ef_search_, k_sz);

    std::vector<idx_t> enter{entry_point_};
    for (int lc = max_level_; lc >= 1; --lc) {
        const auto found = search_layer(query, enter, 1, lc);
        if (found.empty()) {
            break;
        }
        enter.assign(1, found.front().id);
    }

    const auto found = search_layer(query, enter, effective_ef, 0);
    const std::size_t take = std::min(k_sz, found.size());
    for (std::size_t i = 0; i < take; ++i) {
        ids_out[i] = found[i].id;
        distances_out[i] = found[i].distance;
    }
}

SearchResult HNSWIndex::search(const float* query, int k) const {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    SearchResult result;
    result.ids.assign(static_cast<std::size_t>(k), kPadId);
    result.distances.assign(static_cast<std::size_t>(k), kPadDistance);
    search_one(query, k, result.ids.data(), result.distances.data());
    return result;
}

void HNSWIndex::search_batch(
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

std::uint64_t HNSWIndex::graph_digest() const {
    std::uint64_t hash = 14695981039346656037ULL;
    hash = fnv1a64(hash, n_);
    hash = fnv1a64(hash, M_);
    hash = fnv1a64(hash, M0_);
    hash = fnv1a64(hash, seed_);
    hash = fnv1a64_i32(hash, max_level_);
    hash = fnv1a64(hash, static_cast<std::uint64_t>(entry_point_));
    for (std::size_t i = 0; i < n_; ++i) {
        const Node& node = nodes_[i];
        hash = fnv1a64_i32(hash, node.level);
        for (int lc = 0; lc <= node.level; ++lc) {
            const std::vector<idx_t>& neigh = node.neighbors[static_cast<std::size_t>(lc)];
            hash = fnv1a64(hash, neigh.size());
            for (idx_t nb : neigh) {
                hash = fnv1a64(hash, static_cast<std::uint64_t>(nb));
            }
        }
    }
    return hash;
}

void HNSWIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open '" + path + "' for writing");
    }
    out.write(kMagic, 4);
    const std::uint32_t dim = static_cast<std::uint32_t>(dim_);
    const std::uint32_t metric = metric_ == Metric::L2 ? 0u : 1u;
    const std::uint64_t n = n_;
    const std::uint64_t M = M_;
    const std::uint64_t M0 = M0_;
    const std::uint64_t efc = ef_construction_;
    const std::uint64_t efs = ef_search_;
    const std::uint64_t seed = seed_;
    const std::int32_t max_level = max_level_;
    const std::int64_t entry = entry_point_;
    write_pod(out, dim);
    write_pod(out, metric);
    write_pod(out, n);
    write_pod(out, M);
    write_pod(out, M0);
    write_pod(out, efc);
    write_pod(out, efs);
    write_pod(out, seed);
    write_pod(out, max_level);
    write_pod(out, entry);

    if (n_ > 0) {
        const std::size_t vec_bytes =
            checked_mul(vectors_.size(), sizeof(float), "serialized vector bytes");
        out.write(reinterpret_cast<const char*>(vectors_.data()),
                  static_cast<std::streamsize>(vec_bytes));
        for (std::size_t i = 0; i < n_; ++i) {
            const std::int32_t level = nodes_[i].level;
            write_pod(out, level);
        }
        for (std::size_t i = 0; i < n_; ++i) {
            const Node& node = nodes_[i];
            for (int lc = 0; lc <= node.level; ++lc) {
                const std::vector<idx_t>& neigh = node.neighbors[static_cast<std::size_t>(lc)];
                const std::uint32_t degree = static_cast<std::uint32_t>(neigh.size());
                write_pod(out, degree);
                for (idx_t nb : neigh) {
                    const std::int64_t id = nb;
                    write_pod(out, id);
                }
            }
        }
    }
    if (!out) {
        throw std::runtime_error("failed while writing '" + path + "'");
    }
}

void HNSWIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open '" + path + "' for reading");
    }

    char magic[4] = {};
    in.read(magic, 4);
    if (!in || magic[0] != kMagic[0] || magic[1] != kMagic[1] || magic[2] != kMagic[2] ||
        magic[3] != kMagic[3]) {
        throw std::runtime_error("invalid VectorForge HNSW index file: bad magic");
    }

    std::uint32_t dim = 0;
    std::uint32_t metric = 0;
    std::uint64_t n = 0;
    std::uint64_t M = 0;
    std::uint64_t M0 = 0;
    std::uint64_t efc = 0;
    std::uint64_t efs = 0;
    std::uint64_t seed = 0;
    std::int32_t max_level = 0;
    std::int64_t entry = 0;
    read_pod(in, dim, "truncated header");
    read_pod(in, metric, "truncated header");
    read_pod(in, n, "truncated header");
    read_pod(in, M, "truncated header");
    read_pod(in, M0, "truncated header");
    read_pod(in, efc, "truncated header");
    read_pod(in, efs, "truncated header");
    read_pod(in, seed, "truncated header");
    read_pod(in, max_level, "truncated header");
    read_pod(in, entry, "truncated header");

    if (dim == 0 || dim > static_cast<std::uint32_t>(std::numeric_limits<dim_t>::max())) {
        throw std::runtime_error("invalid dim in HNSW index file");
    }
    if (metric > 1u) {
        throw std::runtime_error("invalid metric in HNSW index file");
    }

    const std::uint64_t max_size_t =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (n > max_size_t || M > max_size_t || M0 > max_size_t || efc > max_size_t ||
        efs > max_size_t) {
        throw std::runtime_error("invalid parameter in HNSW index file");
    }
    try {
        require_hnsw_params(static_cast<std::size_t>(M),
                            static_cast<std::size_t>(efc),
                            static_cast<std::size_t>(efs));
    } catch (const std::invalid_argument& ex) {
        throw std::runtime_error(std::string("invalid HNSW parameters in index file: ") +
                                 ex.what());
    }
    if (M0 != 2 * M) {
        throw std::runtime_error("invalid M0 in HNSW index file");
    }

    const std::size_t n_size = static_cast<std::size_t>(n);
    const std::size_t dim_size = static_cast<std::size_t>(dim);
    if (n_size == 0) {
        if (max_level != -1 || entry != -1) {
            throw std::runtime_error("empty HNSW index must have max_level=-1 and entry_point=-1");
        }
    } else {
        if (entry < 0 || static_cast<std::uint64_t>(entry) >= n) {
            throw std::runtime_error("invalid entry_point in HNSW index file");
        }
        if (max_level < 0 || max_level > kMaxLevelCap) {
            throw std::runtime_error("invalid max_level in HNSW index file");
        }
    }

    const std::size_t file_size = static_cast<std::size_t>(std::filesystem::file_size(path));
    const std::size_t header_end = current_stream_pos(in);
    if (header_end > file_size) {
        throw std::runtime_error("invalid VectorForge HNSW index file: header past EOF");
    }

    const std::size_t element_count = checked_mul(n_size, dim_size, "serialized vector count");
    const std::size_t payload_bytes =
        checked_mul(element_count, sizeof(float), "serialized vector bytes");
    const std::size_t remaining_after_header = file_size - header_end;
    if (n_size > 0 && remaining_after_header < payload_bytes) {
        throw std::runtime_error("invalid VectorForge HNSW index file: truncated payload");
    }

    std::vector<float> loaded_vectors(element_count, 0.0f);
    if (n_size > 0) {
        require_read(in,
                     reinterpret_cast<char*>(loaded_vectors.data()),
                     static_cast<std::streamsize>(payload_bytes),
                     "truncated payload");
        validate_finite_block(loaded_vectors.data(), element_count, "serialized vectors");
    }

    std::vector<Node> loaded_nodes(n_size);
    int observed_max = -1;
    if (n_size > 0) {
        for (std::size_t i = 0; i < n_size; ++i) {
            std::int32_t level = 0;
            read_pod(in, level, "truncated node levels");
            if (level < 0 || level > kMaxLevelCap) {
                throw std::runtime_error("invalid node level in HNSW index file");
            }
            loaded_nodes[i].level = level;
            loaded_nodes[i].neighbors.resize(static_cast<std::size_t>(level) + 1);
            observed_max = std::max(observed_max, static_cast<int>(level));
        }
        if (observed_max != max_level) {
            throw std::runtime_error("max_level does not match node levels in HNSW index file");
        }
        const std::size_t entry_i = static_cast<std::size_t>(entry);
        if (loaded_nodes[entry_i].level != max_level) {
            throw std::runtime_error("entry_point is not a max-level node");
        }

        const std::size_t max_deg0 = static_cast<std::size_t>(M0);
        const std::size_t max_deg = static_cast<std::size_t>(M);
        for (std::size_t i = 0; i < n_size; ++i) {
            Node& node = loaded_nodes[i];
            for (int lc = 0; lc <= node.level; ++lc) {
                std::uint32_t degree = 0;
                read_pod(in, degree, "truncated adjacency");
                const std::size_t cap = (lc == 0) ? max_deg0 : max_deg;
                if (static_cast<std::size_t>(degree) > cap) {
                    throw std::runtime_error("degree overflow in HNSW index file");
                }
                if (static_cast<std::size_t>(degree) > n_size) {
                    throw std::runtime_error("impossible neighbor degree in HNSW index file");
                }
                std::vector<idx_t>& neigh = node.neighbors[static_cast<std::size_t>(lc)];
                neigh.resize(degree);
                for (std::uint32_t d = 0; d < degree; ++d) {
                    std::int64_t nb = 0;
                    read_pod(in, nb, "truncated adjacency");
                    if (nb < 0 || static_cast<std::uint64_t>(nb) >= n) {
                        throw std::runtime_error("impossible node id in HNSW adjacency");
                    }
                    if (nb == static_cast<std::int64_t>(i)) {
                        throw std::runtime_error("self-loop in HNSW adjacency");
                    }
                    const Node& other = loaded_nodes[static_cast<std::size_t>(nb)];
                    if (lc > other.level) {
                        throw std::runtime_error("neighbor missing required layer in HNSW graph");
                    }
                    neigh[d] = nb;
                }
                auto sorted = neigh;
                std::sort(sorted.begin(), sorted.end());
                if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
                    throw std::runtime_error("duplicate neighbor in HNSW adjacency");
                }
            }
        }
    }

    const std::size_t consumed = current_stream_pos(in);
    if (consumed < file_size) {
        throw std::runtime_error("invalid VectorForge HNSW index file: unexpected trailing data");
    }
    if (consumed > file_size) {
        throw std::runtime_error("invalid VectorForge HNSW index file: truncated data");
    }

    dim_ = static_cast<dim_t>(dim);
    metric_ = metric == 0u ? Metric::L2 : Metric::Cosine;
    n_ = n_size;
    M_ = static_cast<std::size_t>(M);
    M0_ = static_cast<std::size_t>(M0);
    ef_construction_ = static_cast<std::size_t>(efc);
    ef_search_ = static_cast<std::size_t>(efs);
    seed_ = seed;
    level_mult_ = 1.0 / std::log(static_cast<double>(M_));
    max_level_ = max_level;
    entry_point_ = entry;
    vectors_ = std::move(loaded_vectors);
    nodes_ = std::move(loaded_nodes);
    visit_stamp_.assign(n_, 0);
    visit_gen_ = 1;
    restore_rng_after_n_inserts();
}

}  // namespace vectorforge
