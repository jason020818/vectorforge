#include "vectorforge/flat_index.hpp"
#include "vectorforge/hnsw_index.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using vectorforge::FlatIndex;
using vectorforge::HNSWIndex;
using vectorforge::idx_t;
using vectorforge::Metric;

namespace {

std::vector<float> matrix(std::initializer_list<std::initializer_list<float>> rows) {
    std::vector<float> out;
    for (const auto& row : rows) {
        out.insert(out.end(), row.begin(), row.end());
    }
    return out;
}

std::vector<float> fill_random(std::size_t n, int dim, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> out(n * static_cast<std::size_t>(dim));
    for (float& v : out) {
        const std::uint64_t bits = rng() >> 11;
        const double u = (static_cast<double>(bits) + 0.5) * 0x1p-53;
        v = static_cast<float>(u * 2.0 - 1.0);
    }
    return out;
}

double recall_at_k(const std::vector<idx_t>& ann,
                   const std::vector<idx_t>& gt,
                   std::size_t nq,
                   int k) {
    double hits = 0.0;
    for (std::size_t q = 0; q < nq; ++q) {
        std::set<idx_t> truth;
        for (int i = 0; i < k; ++i) {
            truth.insert(gt[q * static_cast<std::size_t>(k) + static_cast<std::size_t>(i)]);
        }
        for (int i = 0; i < k; ++i) {
            if (truth.count(ann[q * static_cast<std::size_t>(k) + static_cast<std::size_t>(i)]) !=
                0) {
                hits += 1.0;
            }
        }
    }
    return hits / (static_cast<double>(nq) * static_cast<double>(k));
}

HNSWIndex make_index(int dim,
                     const std::string& metric,
                     std::size_t M = 8,
                     std::size_t efc = 32,
                     std::size_t efs = 16,
                     std::uint64_t seed = 42) {
    return HNSWIndex(dim, metric, M, efc, efs, seed);
}

}  // namespace

TEST_CASE("hnsw constructor rejects non-positive dim", "[hnsw]") {
    REQUIRE_THROWS_AS(HNSWIndex(0, Metric::L2), std::invalid_argument);
    REQUIRE_THROWS_AS(HNSWIndex(-1, "cosine"), std::invalid_argument);
}

TEST_CASE("hnsw constructor rejects invalid parameters", "[hnsw]") {
    REQUIRE_THROWS_AS(HNSWIndex(4, "l2", 1, 16, 10, 42), std::invalid_argument);
    REQUIRE_THROWS_AS(HNSWIndex(4, "l2", 16, 8, 10, 42), std::invalid_argument);
    REQUIRE_THROWS_AS(HNSWIndex(4, "l2", 16, 32, 0, 42), std::invalid_argument);
    REQUIRE_THROWS_AS(HNSWIndex(4, "dot"), std::invalid_argument);
}

TEST_CASE("hnsw set_ef_search rejects zero", "[hnsw]") {
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.set_ef_search(0), std::invalid_argument);
    index.set_ef_search(7);
    REQUIRE(index.ef_search() == 7);
}

TEST_CASE("hnsw empty index returns padded results", "[hnsw]") {
    HNSWIndex index(4, "cosine");
    const float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const auto result = index.search(query, 2);
    REQUIRE(result.ids[0] == -1);
    REQUIRE(result.ids[1] == -1);
    REQUIRE(std::isinf(result.distances[0]));
    REQUIRE(index.size() == 0);
    REQUIRE(index.max_level() == -1);
    REQUIRE(index.entry_point() == -1);
}

TEST_CASE("hnsw single vector search", "[hnsw]") {
    HNSWIndex index(2, "l2", 4, 8, 4, 42);
    const float data[] = {1.0f, 2.0f};
    index.add(data, 1);
    REQUIRE(index.size() == 1);
    const float query[] = {1.0f, 2.0f};
    const auto result = index.search(query, 1);
    REQUIRE(result.ids[0] == 0);
    REQUIRE_THAT(result.distances[0], Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("hnsw two vectors L2", "[hnsw][l2]") {
    HNSWIndex index(2, "l2", 4, 8, 8, 42);
    const auto data = matrix({{0.0f, 0.0f}, {10.0f, 0.0f}});
    index.add(data.data(), 2);
    const float query[] = {0.1f, 0.0f};
    const auto result = index.search(query, 2);
    REQUIRE(result.ids[0] == 0);
    REQUIRE(result.ids[1] == 1);
}

TEST_CASE("hnsw k=1 and k=N and k>N", "[hnsw]") {
    HNSWIndex index(1, "l2", 4, 8, 8, 1);
    const float data[] = {5.0f, 1.0f, 3.0f};
    index.add(data, 3);
    const float query[] = {0.0f};

    const auto k1 = index.search(query, 1);
    REQUIRE(k1.ids.size() == 1);
    REQUIRE(k1.ids[0] == 1);

    const auto kn = index.search(query, 3);
    REQUIRE(kn.ids.size() == 3);
    REQUIRE(kn.ids[0] == 1);
    REQUIRE(kn.ids[1] == 2);
    REQUIRE(kn.ids[2] == 0);

    const auto kbig = index.search(query, 5);
    REQUIRE(kbig.ids[0] == 1);
    REQUIRE(kbig.ids[3] == -1);
    REQUIRE(kbig.ids[4] == -1);
    REQUIRE(std::isinf(kbig.distances[3]));
}

TEST_CASE("hnsw k must be positive", "[hnsw]") {
    HNSWIndex index(1, "l2");
    const float q[] = {0.0f};
    REQUIRE_THROWS_AS(index.search(q, 0), std::invalid_argument);
}

TEST_CASE("hnsw duplicate vectors break ties by smaller id", "[hnsw][deterministic]") {
    HNSWIndex index(2, "l2", 8, 16, 16, 42);
    const auto data = matrix({{1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}});
    index.add(data.data(), 3);
    const float query[] = {1.0f, 1.0f};
    const auto result = index.search(query, 3);
    REQUIRE(result.ids[0] == 0);
    REQUIRE(result.ids[1] == 1);
    REQUIRE(result.ids[2] == 2);
    REQUIRE(result.distances[0] == 0.0f);
}

TEST_CASE("hnsw zero vectors use the same cosine semantics as FlatIndex", "[hnsw][zero][cosine]") {
    HNSWIndex hnsw(3, "cosine", 4, 8, 8, 42);
    FlatIndex flat(3, "cosine");
    const auto data = matrix({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}});
    hnsw.add(data.data(), 3);
    flat.add(data.data(), 3);
    const float query[] = {0.0f, 0.0f, 0.0f};
    const auto h = hnsw.search(query, 3);
    const auto f = flat.search(query, 3);
    REQUIRE(h.ids == f.ids);
    REQUIRE_THAT(h.distances[0], Catch::Matchers::WithinAbs(f.distances[0], 1e-6f));
}

TEST_CASE("hnsw cosine ranks like FlatIndex on a tiny set", "[hnsw][cosine]") {
    HNSWIndex index(2, "cosine", 4, 8, 8, 42);
    const auto data = matrix({{1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}});
    index.add(data.data(), 3);
    const float query[] = {1.0f, 0.0f};
    const auto result = index.search(query, 3);
    REQUIRE(result.ids[0] == 0);
    REQUIRE(result.ids[1] == 1);
    REQUIRE(result.ids[2] == 2);
}

TEST_CASE("hnsw batch search matches per-query search", "[hnsw][batch]") {
    HNSWIndex index(2, "l2", 4, 8, 8, 42);
    const auto data = matrix({{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}});
    index.add(data.data(), 3);
    const auto queries = matrix({{0.1f, 0.0f}, {0.0f, 0.9f}});
    std::vector<idx_t> ids(4, -2);
    std::vector<float> dists(4, -2.0f);
    index.search_batch(queries.data(), 2, 2, ids.data(), dists.data());
    const auto r0 = index.search(queries.data(), 2);
    const auto r1 = index.search(queries.data() + 2, 2);
    REQUIRE(ids[0] == r0.ids[0]);
    REQUIRE(ids[1] == r0.ids[1]);
    REQUIRE(ids[2] == r1.ids[0]);
    REQUIRE(ids[3] == r1.ids[1]);
}

TEST_CASE("hnsw rejects non-finite add and search", "[hnsw][validation]") {
    for (const auto metric : {"l2", "cosine"}) {
        HNSWIndex index(2, metric, 4, 8, 8, 42);
        std::vector<float> with_nan{0.0f, std::numeric_limits<float>::quiet_NaN()};
        std::vector<float> with_pos_inf{0.0f, std::numeric_limits<float>::infinity()};
        std::vector<float> with_neg_inf{0.0f, -std::numeric_limits<float>::infinity()};
        REQUIRE_THROWS_AS(index.add(with_nan.data(), 1), std::invalid_argument);
        REQUIRE_THROWS_AS(index.add(with_pos_inf.data(), 1), std::invalid_argument);
        REQUIRE_THROWS_AS(index.add(with_neg_inf.data(), 1), std::invalid_argument);

        const float data[] = {0.0f, 0.0f, 1.0f, 1.0f};
        index.add(data, 2);
        REQUIRE_THROWS_AS(index.search(with_nan.data(), 1), std::invalid_argument);
        REQUIRE_THROWS_AS(index.search(with_pos_inf.data(), 1), std::invalid_argument);
        REQUIRE_THROWS_AS(index.search(with_neg_inf.data(), 1), std::invalid_argument);
    }
}

TEST_CASE("hnsw batch search rejects non-finite queries", "[hnsw][validation][batch]") {
    HNSWIndex index(2, "l2", 4, 8, 8, 42);
    const float data[] = {0.0f, 0.0f, 1.0f, 1.0f};
    index.add(data, 2);
    std::vector<float> queries{0.0f, 0.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f};
    std::vector<idx_t> ids(4, -1);
    std::vector<float> dists(4, 0.0f);
    REQUIRE_THROWS_AS(index.search_batch(queries.data(), 2, 2, ids.data(), dists.data()),
                      std::invalid_argument);
}

TEST_CASE("hnsw random small dataset is exact vs FlatIndex", "[hnsw]") {
    const int dim = 8;
    const std::size_t n = 40;
    const auto data = fill_random(n, dim, 7);
    HNSWIndex hnsw(dim, "l2", 12, 40, 40, 99);
    FlatIndex flat(dim, "l2");
    hnsw.add(data.data(), n);
    flat.add(data.data(), n);
    const auto queries = fill_random(6, dim, 11);
    for (std::size_t q = 0; q < 6; ++q) {
        const auto h = hnsw.search(queries.data() + q * static_cast<std::size_t>(dim), 5);
        const auto f = flat.search(queries.data() + q * static_cast<std::size_t>(dim), 5);
        REQUIRE(h.ids == f.ids);
    }
}

TEST_CASE("hnsw same seed reproduces graph and search", "[hnsw][deterministic]") {
    const int dim = 12;
    const std::size_t n = 80;
    const auto data = fill_random(n, dim, 123);
    auto a = make_index(dim, "l2", 8, 32, 24, 42);
    auto b = make_index(dim, "l2", 8, 32, 24, 42);
    a.add(data.data(), n);
    b.add(data.data(), n);
    REQUIRE(a.graph_digest() == b.graph_digest());
    REQUIRE(a.max_level() == b.max_level());
    REQUIRE(a.entry_point() == b.entry_point());
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(a.node_level(static_cast<idx_t>(i)) == b.node_level(static_cast<idx_t>(i)));
        for (int lc = 0; lc <= a.node_level(static_cast<idx_t>(i)); ++lc) {
            REQUIRE(a.neighbors(static_cast<idx_t>(i), lc) ==
                    b.neighbors(static_cast<idx_t>(i), lc));
        }
    }
    const auto query = fill_random(1, dim, 5);
    const auto ra = a.search(query.data(), 10);
    const auto rb = b.search(query.data(), 10);
    REQUIRE(ra.ids == rb.ids);
    REQUIRE(ra.distances == rb.distances);
}

TEST_CASE("hnsw different seeds can produce different graphs", "[hnsw][deterministic]") {
    const int dim = 8;
    const std::size_t n = 120;
    const auto data = fill_random(n, dim, 3);
    auto a = make_index(dim, "cosine", 8, 32, 16, 1);
    auto b = make_index(dim, "cosine", 8, 32, 16, 2);
    a.add(data.data(), n);
    b.add(data.data(), n);
    bool levels_differ = false;
    for (std::size_t i = 0; i < n; ++i) {
        if (a.node_level(static_cast<idx_t>(i)) != b.node_level(static_cast<idx_t>(i))) {
            levels_differ = true;
            break;
        }
    }
    REQUIRE(levels_differ);
    REQUIRE(a.graph_digest() != b.graph_digest());
}

TEST_CASE("hnsw degree limits and neighbor ids are valid", "[hnsw]") {
    const int dim = 6;
    const std::size_t n = 200;
    const std::size_t M = 6;
    const auto data = fill_random(n, dim, 21);
    HNSWIndex index(dim, "l2", M, 24, 16, 42);
    index.add(data.data(), n);
    REQUIRE(index.M0() == 2 * M);
    for (std::size_t i = 0; i < n; ++i) {
        const int level = index.node_level(static_cast<idx_t>(i));
        REQUIRE(level >= 0);
        for (int lc = 0; lc <= level; ++lc) {
            const auto neigh = index.neighbors(static_cast<idx_t>(i), lc);
            const std::size_t cap = (lc == 0) ? index.M0() : index.M();
            REQUIRE(neigh.size() <= cap);
            std::set<idx_t> uniq;
            for (idx_t nb : neigh) {
                REQUIRE(nb >= 0);
                REQUIRE(static_cast<std::size_t>(nb) < n);
                REQUIRE(nb != static_cast<idx_t>(i));
                REQUIRE(index.node_level(nb) >= lc);
                REQUIRE(uniq.insert(nb).second);
            }
        }
    }
}

TEST_CASE("hnsw effective ef is at least k", "[hnsw]") {
    const int dim = 8;
    const std::size_t n = 50;
    const auto data = fill_random(n, dim, 9);
    HNSWIndex index(dim, "l2", 8, 32, 4, 42);
    index.add(data.data(), n);
    FlatIndex flat(dim, "l2");
    flat.add(data.data(), n);
    const auto query = fill_random(1, dim, 4);
    const auto h = index.search(query.data(), 20);
    const auto f = flat.search(query.data(), 20);
    REQUIRE(h.ids.size() == 20);
    REQUIRE(h.ids == f.ids);
}

TEST_CASE("hnsw synthetic Recall@10 gate vs FlatIndex", "[hnsw][recall]") {
    constexpr int dim = 64;
    constexpr std::size_t n = 10000;
    constexpr std::size_t nq = 100;
    constexpr int k10 = 10;
    constexpr int k100 = 100;
    const auto data = fill_random(n, dim, 42);
    const auto queries = fill_random(nq, dim, 43);

    FlatIndex flat(dim, "l2");
    flat.add(data.data(), n);
    HNSWIndex hnsw(dim, "l2", 16, 200, 100, 42);
    hnsw.add(data.data(), n);

    std::vector<idx_t> h10(nq * static_cast<std::size_t>(k10));
    std::vector<float> hd10(nq * static_cast<std::size_t>(k10));
    std::vector<idx_t> f10(nq * static_cast<std::size_t>(k10));
    std::vector<float> fd10(nq * static_cast<std::size_t>(k10));
    hnsw.search_batch(queries.data(), nq, k10, h10.data(), hd10.data());
    flat.search_batch(queries.data(), nq, k10, f10.data(), fd10.data());

    std::vector<idx_t> h100(nq * static_cast<std::size_t>(k100));
    std::vector<float> hd100(nq * static_cast<std::size_t>(k100));
    std::vector<idx_t> f100(nq * static_cast<std::size_t>(k100));
    std::vector<float> fd100(nq * static_cast<std::size_t>(k100));
    hnsw.search_batch(queries.data(), nq, k100, h100.data(), hd100.data());
    flat.search_batch(queries.data(), nq, k100, f100.data(), fd100.data());

    const double r10 = recall_at_k(h10, f10, nq, k10);
    const double r100 = recall_at_k(h100, f100, nq, k100);
    INFO("Recall@10=" << r10 << " Recall@100=" << r100);
    REQUIRE(r10 >= 0.90);
    REQUIRE(r100 >= 0.0);
    REQUIRE(r100 <= 1.0);
}
