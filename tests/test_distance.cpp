#include "vectorforge/distance.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

using vectorforge::Metric;
using vectorforge::distance::compute;
using vectorforge::distance::cosine_distance;
using vectorforge::distance::cosine_similarity;
using vectorforge::distance::l2;

TEST_CASE("L2 of identical vectors is zero", "[distance][l2]") {
    const std::vector<float> a{1.0f, -2.0f, 3.5f, 0.0f};
    REQUIRE(l2(a.data(), a.data(), 4) == 0.0f);
}

TEST_CASE("L2 matches the Euclidean formula", "[distance][l2]") {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, 6.0f, 3.0f};
    const float expected = std::sqrt(9.0f + 16.0f + 0.0f);
    REQUIRE_THAT(l2(a.data(), b.data(), 3), Catch::Matchers::WithinAbs(expected, 1e-6f));
}

TEST_CASE("cosine similarity of a vector with itself is 1", "[distance][cosine]") {
    const std::vector<float> a{0.3f, -1.2f, 4.0f};
    REQUIRE_THAT(cosine_similarity(a.data(), a.data(), 3), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
}

TEST_CASE("orthogonal vectors have cosine similarity 0", "[distance][cosine]") {
    const std::vector<float> a{1.0f, 0.0f};
    const std::vector<float> b{0.0f, 1.0f};
    REQUIRE_THAT(cosine_similarity(a.data(), b.data(), 2), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("zero vector cosine similarity is defined as 0", "[distance][cosine][zero]") {
    const std::vector<float> z{0.0f, 0.0f, 0.0f};
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    REQUIRE(cosine_similarity(z.data(), a.data(), 3) == 0.0f);
    REQUIRE(cosine_similarity(z.data(), z.data(), 3) == 0.0f);
    REQUIRE(cosine_distance(z.data(), a.data(), 3) == 1.0f);
}

TEST_CASE("compute dispatches L2 and cosine", "[distance]") {
    const std::vector<float> a{1.0f, 0.0f};
    const std::vector<float> b{0.0f, 1.0f};
    REQUIRE_THAT(compute(Metric::L2, a.data(), b.data(), 2),
                 Catch::Matchers::WithinAbs(std::sqrt(2.0f), 1e-6f));
    REQUIRE_THAT(compute(Metric::Cosine, a.data(), b.data(), 2),
                 Catch::Matchers::WithinAbs(1.0f, 1e-6f));
}
