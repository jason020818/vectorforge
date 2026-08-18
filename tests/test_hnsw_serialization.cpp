#include "vectorforge/hnsw_index.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using vectorforge::HNSWIndex;

namespace {

std::filesystem::path temp_file(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
}

void write_vh01_header(std::ofstream& out,
                       std::uint32_t dim,
                       std::uint32_t metric,
                       std::uint64_t n,
                       std::uint64_t M,
                       std::uint64_t M0,
                       std::uint64_t efc,
                       std::uint64_t efs,
                       std::uint64_t seed,
                       std::int32_t max_level,
                       std::int64_t entry) {
    out.write("VH01", 4);
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
}

HNSWIndex populated(const char* metric) {
    HNSWIndex index(2, metric, 4, 8, 8, 42);
    const float data[] = {0.0f, 0.0f, 3.0f, 4.0f, 1.0f, 1.0f, 8.0f, -1.0f};
    index.add(data, 4);
    return index;
}

}  // namespace

TEST_CASE("hnsw save then load preserves graph and search for L2", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_l2.bin");
    HNSWIndex index = populated("l2");
    index.save(path.string());

    HNSWIndex loaded(8, "cosine", 16, 32, 16, 7);
    loaded.load(path.string());
    REQUIRE(loaded.dim() == 2);
    REQUIRE(loaded.metric() == vectorforge::Metric::L2);
    REQUIRE(loaded.size() == 4);
    REQUIRE(loaded.M() == 4);
    REQUIRE(loaded.ef_construction() == 8);
    REQUIRE(loaded.ef_search() == 8);
    REQUIRE(loaded.seed() == 42);
    REQUIRE(loaded.max_level() == index.max_level());
    REQUIRE(loaded.entry_point() == index.entry_point());
    REQUIRE(loaded.graph_digest() == index.graph_digest());

    const float query[] = {0.0f, 0.0f};
    const auto before = index.search(query, 3);
    const auto after = loaded.search(query, 3);
    REQUIRE(after.ids == before.ids);
    REQUIRE_THAT(after.distances[0], Catch::Matchers::WithinAbs(before.distances[0], 0.0f));
    std::filesystem::remove(path);
}

TEST_CASE("hnsw save then load preserves cosine search", "[hnsw][serialization][cosine]") {
    const auto path = temp_file("vectorforge_hnsw_cosine.bin");
    HNSWIndex index = populated("cosine");
    index.save(path.string());
    HNSWIndex loaded(2, "l2");
    loaded.load(path.string());
    REQUIRE(loaded.metric() == vectorforge::Metric::Cosine);
    REQUIRE(loaded.graph_digest() == index.graph_digest());
    const float query[] = {1.0f, 0.0f};
    REQUIRE(loaded.search(query, 2).ids == index.search(query, 2).ids);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw empty save/load round trip", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_empty.bin");
    HNSWIndex index(4, "l2", 8, 16, 10, 99);
    index.save(path.string());
    HNSWIndex loaded(2, "cosine");
    loaded.load(path.string());
    REQUIRE(loaded.size() == 0);
    REQUIRE(loaded.dim() == 4);
    REQUIRE(loaded.M() == 8);
    REQUIRE(loaded.seed() == 99);
    REQUIRE(loaded.max_level() == -1);
    REQUIRE(loaded.entry_point() == -1);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects bad magic", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_bad_magic.bin");
    {
        std::ofstream out(path, std::ios::binary);
        out.write("VF01", 4);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects truncated header", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_trunc_header.bin");
    {
        std::ofstream out(path, std::ios::binary);
        out.write("VH01", 4);
        const std::uint32_t dim = 2;
        write_pod(out, dim);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects invalid metric", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_bad_metric.bin");
    {
        std::ofstream out(path, std::ios::binary);
        write_vh01_header(out, 2, 9, 0, 4, 8, 8, 8, 42, -1, -1);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects invalid M", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_bad_m.bin");
    {
        std::ofstream out(path, std::ios::binary);
        write_vh01_header(out, 2, 0, 0, 1, 2, 8, 8, 42, -1, -1);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects truncated payload", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_trunc_payload.bin");
    {
        std::ofstream out(path, std::ios::binary);
        write_vh01_header(out, 2, 0, 2, 4, 8, 8, 8, 42, 0, 0);
        const float one[2] = {1.0f, 2.0f};
        out.write(reinterpret_cast<const char*>(one), sizeof(one));
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects trailing bytes", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_trailing.bin");
    populated("l2").save(path.string());
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        const char extra = 0x7f;
        out.write(&extra, 1);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects NaN vectors", "[hnsw][serialization][validation]") {
    const auto path = temp_file("vectorforge_hnsw_nan.bin");
    {
        std::ofstream out(path, std::ios::binary);
        write_vh01_header(out, 2, 0, 1, 4, 8, 8, 8, 42, 0, 0);
        const float payload[] = {1.0f, std::numeric_limits<float>::quiet_NaN()};
        out.write(reinterpret_cast<const char*>(payload), sizeof(payload));
        const std::int32_t level = 0;
        write_pod(out, level);
        const std::uint32_t degree = 0;
        write_pod(out, degree);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::invalid_argument);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects Inf vectors", "[hnsw][serialization][validation]") {
    const auto path = temp_file("vectorforge_hnsw_inf.bin");
    {
        std::ofstream out(path, std::ios::binary);
        write_vh01_header(out, 2, 0, 1, 4, 8, 8, 8, 42, 0, 0);
        const float payload[] = {0.0f, std::numeric_limits<float>::infinity()};
        out.write(reinterpret_cast<const char*>(payload), sizeof(payload));
        const std::int32_t level = 0;
        write_pod(out, level);
        const std::uint32_t degree = 0;
        write_pod(out, degree);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::invalid_argument);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects impossible neighbor ids", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_bad_nb.bin");
    {
        std::ofstream out(path, std::ios::binary);
        write_vh01_header(out, 2, 0, 1, 4, 8, 8, 8, 42, 0, 0);
        const float payload[] = {1.0f, 2.0f};
        out.write(reinterpret_cast<const char*>(payload), sizeof(payload));
        const std::int32_t level = 0;
        write_pod(out, level);
        const std::uint32_t degree = 1;
        write_pod(out, degree);
        const std::int64_t nb = 99;
        write_pod(out, nb);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("hnsw load rejects degree overflow", "[hnsw][serialization]") {
    const auto path = temp_file("vectorforge_hnsw_deg.bin");
    {
        std::ofstream out(path, std::ios::binary);
        write_vh01_header(out, 2, 0, 1, 4, 8, 8, 8, 42, 0, 0);
        const float payload[] = {1.0f, 2.0f};
        out.write(reinterpret_cast<const char*>(payload), sizeof(payload));
        const std::int32_t level = 0;
        write_pod(out, level);
        const std::uint32_t degree = 9;  // M0 = 8
        write_pod(out, degree);
    }
    HNSWIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("failed hnsw load preserves existing state", "[hnsw][serialization][safety]") {
    HNSWIndex index = populated("l2");
    const float query[] = {0.0f, 0.0f};
    const auto before = index.search(query, 2);
    const auto digest = index.graph_digest();

    const auto path = temp_file("vectorforge_hnsw_corrupt_preserve.bin");
    {
        std::ofstream out(path, std::ios::binary);
        write_vh01_header(out, 2, 1, 1, 4, 8, 8, 8, 1, 0, 0);
        const float payload[] = {1.0f, std::numeric_limits<float>::quiet_NaN()};
        out.write(reinterpret_cast<const char*>(payload), sizeof(payload));
        const std::int32_t level = 0;
        write_pod(out, level);
        const std::uint32_t degree = 0;
        write_pod(out, degree);
    }
    REQUIRE_THROWS_AS(index.load(path.string()), std::invalid_argument);
    REQUIRE(index.dim() == 2);
    REQUIRE(index.metric() == vectorforge::Metric::L2);
    REQUIRE(index.size() == 4);
    REQUIRE(index.graph_digest() == digest);
    const auto after = index.search(query, 2);
    REQUIRE(after.ids == before.ids);
    std::filesystem::remove(path);
}

TEST_CASE("failed trailing hnsw load preserves state", "[hnsw][serialization][safety]") {
    HNSWIndex index = populated("cosine");
    const float query[] = {1.0f, 0.0f};
    const auto before = index.search(query, 2);

    const auto path = temp_file("vectorforge_hnsw_trail_preserve.bin");
    populated("l2").save(path.string());
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        const char extra = 0x01;
        out.write(&extra, 1);
    }
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    REQUIRE(index.metric() == vectorforge::Metric::Cosine);
    REQUIRE(index.size() == 4);
    REQUIRE(index.search(query, 2).ids == before.ids);
    std::filesystem::remove(path);
}
