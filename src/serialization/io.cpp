#include "vectorforge/distance.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace vectorforge::io {

// Little-endian binary helpers used by index save/load. Isolated so later
// formats (HNSW) can share the same primitives without depending on FlatIndex.

inline void write_bytes(std::ostream& out, const void* data, std::size_t n) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!out) {
        throw std::runtime_error("write failed");
    }
}

inline void read_bytes(std::istream& in, void* data, std::size_t n) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(n));
    if (!in) {
        throw std::runtime_error("read failed");
    }
}

}  // namespace vectorforge::io
