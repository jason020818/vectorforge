#pragma once

#include "vectorforge/index.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace vectorforge {

// Hierarchical NSW approximate nearest neighbor index (Malkov & Yashunin, 2018).
//
// Phase 2 baseline: paper-faithful, deterministic, scalar FP32. Not a
// performance implementation — no SIMD, quantization, or parallel construction.
//
// Graph degree limits:
//   upper layers: M
//   layer 0:      M0 = 2 * M
//
// Level generation uses a seeded std::mt19937_64. Each inserted vector consumes
// one uniform U in (0, 1) from 53-bit mantissa bits:
//   U = (floor(x / 2^11) + 0.5) * 2^-53
//   level = floor(-ln(U) / ln(M))
// There is no wall-clock or random_device dependence.
//
// Neighbor selection is paper Algorithm 4 (heuristic) with
// extendCandidates = false and keepPrunedConnections = true: candidates that
// fail the diversity test are retained only to fill remaining slots up to the
// layer degree cap, in (distance, id) order.
class HNSWIndex final : public Index {
public:
    static constexpr std::size_t kDefaultM = 16;
    static constexpr std::size_t kDefaultEfConstruction = 200;
    static constexpr std::size_t kDefaultEfSearch = 100;
    static constexpr std::uint64_t kDefaultSeed = 42;

    HNSWIndex(dim_t dim,
              Metric metric,
              std::size_t M = kDefaultM,
              std::size_t ef_construction = kDefaultEfConstruction,
              std::size_t ef_search = kDefaultEfSearch,
              std::uint64_t seed = kDefaultSeed);

    HNSWIndex(dim_t dim,
              const std::string& metric,
              std::size_t M = kDefaultM,
              std::size_t ef_construction = kDefaultEfConstruction,
              std::size_t ef_search = kDefaultEfSearch,
              std::uint64_t seed = kDefaultSeed);

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

    void set_ef_search(std::size_t ef);

    [[nodiscard]] std::size_t M() const { return M_; }
    [[nodiscard]] std::size_t M0() const { return M0_; }
    [[nodiscard]] std::size_t ef_construction() const { return ef_construction_; }
    [[nodiscard]] std::size_t ef_search() const { return ef_search_; }
    [[nodiscard]] std::uint64_t seed() const { return seed_; }
    [[nodiscard]] int max_level() const { return max_level_; }
    [[nodiscard]] idx_t entry_point() const { return entry_point_; }

    [[nodiscard]] int node_level(idx_t id) const;
    [[nodiscard]] std::vector<idx_t> neighbors(idx_t id, int layer) const;

    // Stable digest of node levels and ordered adjacency lists. Does not
    // include pointers or process-specific data.
    [[nodiscard]] std::uint64_t graph_digest() const;

private:
    struct Node {
        int level = 0;
        std::vector<std::vector<idx_t>> neighbors;
    };

    struct Candidate {
        float distance = 0.0f;
        idx_t id = -1;
    };

    void search_one(const float* query, int k, idx_t* ids_out, float* distances_out) const;
    void insert_node(idx_t id);
    [[nodiscard]] int random_level();
    [[nodiscard]] float dist(const float* query, idx_t id) const;
    [[nodiscard]] const float* vec(idx_t id) const;

    [[nodiscard]] std::vector<Candidate> search_layer(const float* query,
                                                      const std::vector<idx_t>& entries,
                                                      std::size_t ef,
                                                      int layer) const;

    [[nodiscard]] std::vector<idx_t> select_neighbors_heuristic(std::vector<Candidate> candidates,
                                                                std::size_t m) const;

    void prune_neighbors(idx_t id, int layer, std::size_t max_degree);
    void restore_rng_after_n_inserts();

    dim_t dim_;
    Metric metric_;
    std::size_t n_{0};
    std::size_t M_;
    std::size_t M0_;
    std::size_t ef_construction_;
    std::size_t ef_search_;
    std::uint64_t seed_;
    double level_mult_;
    int max_level_{-1};
    idx_t entry_point_{-1};
    std::mt19937_64 rng_;
    std::vector<float> vectors_;
    std::vector<Node> nodes_;
    mutable std::vector<std::uint32_t> visit_stamp_;
    mutable std::uint32_t visit_gen_{1};
};

}  // namespace vectorforge
