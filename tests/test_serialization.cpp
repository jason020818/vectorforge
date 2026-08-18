#include "vectorforge/flat_index.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using vectorforge::FlatIndex;

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
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "vectorforge_truncated.bin";
    {
        std::ofstream out(path, std::ios::binary);
        out.write("VF01", 4);
    }
    FlatIndex index(2, "l2");
    REQUIRE_THROWS_AS(index.load(path.string()), std::runtime_error);
    std::filesystem::remove(path);
}
