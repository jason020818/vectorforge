#include "vectorforge/flat_index.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using vectorforge::FlatIndex;

namespace {

void write_raw_header(std::ofstream& out,
                      const char magic[4],
                      std::uint32_t kind,
                      std::uint32_t dim,
                      std::uint32_t metric,
                      std::uint64_t n) {
    out.write(magic, 4);
    out.write(reinterpret_cast<const char*>(&kind), sizeof(kind));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
}

std::filesystem::path temp_file(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

}  // namespace

TEST_CASE("save then load preserves dim, metric, size, and search", "[serialization]") {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vectorforge_flat_roundtrip.bin";

    FlatIndex index(2, "l2");
    const float data[] = {0.0f, 0.0f, 3.0f, 4.0f, 1.0f, 1.0f};
    index.add(data, 3);
    index.save(path.string());

    FlatIndex loaded(8, "cosine");
    loaded.load(path.string());
    REQUIRE(loaded.dim() == 2);
    REQUIRE(loaded.metric() == vectorforge::Metric::L2);
    REQUIRE(loaded.size() == 3);

    const float query[] = {0.0f, 0.0f};
    const auto before = index.search(query, 2);
    const auto after = loaded.search(query, 2);
    REQUIRE(after.ids == before.ids);
    REQUIRE_THAT(after.distances[0], Catch::Matchers::WithinAbs(before.distances[0], 0.0f));
    REQUIRE_THAT(after.distances[1], Catch::Matchers::WithinAbs(before.distances[1], 1e-6f));

    std::filesystem::remove(path);
}

TEST_CASE("load rejects a truncated file", "[serialization]") {
    const std::filesystem::path path = temp_file("vectorforge_truncated.bin");
    {
        std::ofstream out(path, std::ios::binary);
        out.write("VF01", 4);
    }
    FlatIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("load rejects invalid metric", "[serialization]") {
    const std::filesystem::path path = temp_file("vectorforge_bad_metric.bin");
    {
        std::ofstream out(path, std::ios::binary);
        const char magic[4] = {'V', 'F', '0', '1'};
        write_raw_header(out, magic, 1, 2, 9, 0);
    }
    FlatIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("load rejects invalid kind", "[serialization]") {
    const std::filesystem::path path = temp_file("vectorforge_bad_kind.bin");
    {
        std::ofstream out(path, std::ios::binary);
        const char magic[4] = {'V', 'F', '0', '1'};
        write_raw_header(out, magic, 99, 2, 0, 0);
    }
    FlatIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}

TEST_CASE("load rejects corrupt size overflow", "[serialization]") {
    const std::filesystem::path path = temp_file("vectorforge_overflow_size.bin");
    {
        std::ofstream out(path, std::ios::binary);
        const char magic[4] = {'V', 'F', '0', '1'};
        write_raw_header(
            out,
            magic,
            1,
            2,
            0,
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 2 + 1));
    }
    FlatIndex index(2, "l2");
    REQUIRE_THROWS(index.load(path.string()));
    std::filesystem::remove(path);
}

TEST_CASE("load rejects corrupt byte count overflow", "[serialization]") {
    const std::filesystem::path path = temp_file("vectorforge_overflow_bytes.bin");
    {
        std::ofstream out(path, std::ios::binary);
        const char magic[4] = {'V', 'F', '0', '1'};
        write_raw_header(out,
                         magic,
                         1,
                         std::numeric_limits<std::uint32_t>::max(),
                         0,
                         std::numeric_limits<std::uint64_t>::max());
    }
    FlatIndex index(2, "l2");
    REQUIRE_THROWS(index.load(path.string()));
    std::filesystem::remove(path);
}

TEST_CASE("load rejects truncated payload after valid header", "[serialization]") {
    const std::filesystem::path path = temp_file("vectorforge_truncated_payload.bin");
    {
        std::ofstream out(path, std::ios::binary);
        const char magic[4] = {'V', 'F', '0', '1'};
        write_raw_header(out, magic, 1, 2, 0, 2);
        const float one_vector[2] = {1.0f, 2.0f};
        out.write(reinterpret_cast<const char*>(one_vector), sizeof(one_vector));
    }
    FlatIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}
