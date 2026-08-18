#include "vectorforge/flat_index.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

using vectorforge::FlatIndex;
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

}  // namespace

TEST_CASE("constructor rejects non-positive dim", "[flat]") {
    REQUIRE_THROWS_AS(FlatIndex(0, Metric::L2), std::invalid_argument);
    REQUIRE_THROWS_AS(FlatIndex(-1, "cosine"), std::invalid_argument);
}

TEST_CASE("unknown metric is rejected", "[flat]") {
    REQUIRE_THROWS_AS(FlatIndex(4, "dot"), std::invalid_argument);
}

TEST_CASE("malformed add is rejected", "[flat]") {
    FlatIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.add(nullptr, 1), std::invalid_argument);
}

TEST_CASE("L2 k=1 returns the nearest neighbor", "[flat][l2]") {
    FlatIndex index(2, "l2");
    const auto data = matrix({{0.0f, 0.0f}, {1.0f, 0.0f}, {10.0f, 10.0f}});
    index.add(data.data(), 3);
    REQUIRE(index.size() == 3);

    const float query[] = {0.9f, 0.1f};
    const auto result = index.search(query, 1);
    REQUIRE(result.ids[0] == 1);
    REQUIRE_THAT(result.distances[0],
                 Catch::Matchers::WithinAbs(std::sqrt(0.1f * 0.1f + 0.1f * 0.1f), 1e-5f));
}

TEST_CASE("k equals N returns every vector, nearest first", "[flat]") {
    FlatIndex index(1, "l2");
    const float data[] = {5.0f, 1.0f, 3.0f};
    index.add(data, 3);
    const float query[] = {0.0f};
    const auto result = index.search(query, 3);
    REQUIRE(result.ids.size() == 3);
    REQUIRE(result.ids[0] == 1);
    REQUIRE(result.ids[1] == 2);
    REQUIRE(result.ids[2] == 0);
}

TEST_CASE("duplicate vectors break ties by smaller id", "[flat][deterministic]") {
    FlatIndex index(2, "l2");
    const auto data = matrix({{1.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 1.0f}});
    index.add(data.data(), 3);
    const float query[] = {1.0f, 1.0f};
    const auto result = index.search(query, 3);
    REQUIRE(result.ids[0] == 0);
    REQUIRE(result.ids[1] == 1);
    REQUIRE(result.ids[2] == 2);
    REQUIRE(result.distances[0] == 0.0f);
}

TEST_CASE("zero vector is a valid database entry and query", "[flat][zero]") {
    FlatIndex index(3, "l2");
    const auto data = matrix({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}});
    index.add(data.data(), 2);
    const float query[] = {0.0f, 0.0f, 0.0f};
    const auto result = index.search(query, 1);
    REQUIRE(result.ids[0] == 0);
    REQUIRE(result.distances[0] == 0.0f);
}

TEST_CASE("cosine search ranks by cosine distance", "[flat][cosine]") {
    FlatIndex index(2, "cosine");
    const auto data = matrix({{1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}});
    index.add(data.data(), 3);
    const float query[] = {1.0f, 0.0f};
    const auto result = index.search(query, 3);
    REQUIRE(result.ids[0] == 0);
    REQUIRE(result.ids[1] == 1);
    REQUIRE(result.ids[2] == 2);
}

TEST_CASE("k larger than N pads with -1 / inf", "[flat]") {
    FlatIndex index(1, "l2");
    const float data[] = {1.0f};
    index.add(data, 1);
    const float query[] = {0.0f};
    const auto result = index.search(query, 4);
    REQUIRE(result.ids[0] == 0);
    REQUIRE(result.ids[1] == -1);
    REQUIRE(result.ids[2] == -1);
    REQUIRE(result.ids[3] == -1);
    REQUIRE(std::isinf(result.distances[1]));
}

TEST_CASE("k must be positive", "[flat]") {
    FlatIndex index(1, "l2");
    const float q[] = {0.0f};
    REQUIRE_THROWS_AS(index.search(q, 0), std::invalid_argument);
}

TEST_CASE("batch search matches per-query search", "[flat][batch]") {
    FlatIndex index(2, "l2");
    const auto data = matrix({{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}});
    index.add(data.data(), 3);

    const auto queries = matrix({{0.1f, 0.0f}, {0.0f, 0.9f}});
    std::vector<idx_t> ids(2 * 2, -2);
    std::vector<float> dists(2 * 2, -2.0f);
    index.search_batch(queries.data(), 2, 2, ids.data(), dists.data());

    const auto r0 = index.search(queries.data(), 2);
    const auto r1 = index.search(queries.data() + 2, 2);
    REQUIRE(ids[0] == r0.ids[0]);
    REQUIRE(ids[1] == r0.ids[1]);
    REQUIRE(ids[2] == r1.ids[0]);
    REQUIRE(ids[3] == r1.ids[1]);
}

TEST_CASE("empty index returns padded results", "[flat]") {
    FlatIndex index(4, "cosine");
    const float query[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const auto result = index.search(query, 2);
    REQUIRE(result.ids[0] == -1);
    REQUIRE(result.ids[1] == -1);
}
