#include "vectorforge/flat_index.hpp"
#include "vectorforge/hnsw_index.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using F32C = py::array_t<float, py::array::c_style | py::array::forcecast>;

void check_dim(int index_dim, py::ssize_t dim) {
    if (dim != static_cast<py::ssize_t>(index_dim)) {
        throw py::value_error("expected last dimension " + std::to_string(index_dim) + ", got " +
                              std::to_string(dim));
    }
}

template <typename Index>
py::tuple search_numpy(const Index& self, F32C queries, int k) {
    const auto buf = queries.request();
    bool squeeze = false;
    std::size_t nq = 0;
    const float* ptr = static_cast<const float*>(buf.ptr);
    if (buf.ndim == 1) {
        check_dim(self.dim(), buf.shape[0]);
        squeeze = true;
        nq = 1;
    } else if (buf.ndim == 2) {
        check_dim(self.dim(), buf.shape[1]);
        nq = static_cast<std::size_t>(buf.shape[0]);
    } else {
        throw py::value_error("queries must have shape (dim,) or (n, dim)");
    }
    if (k <= 0) {
        throw py::value_error("k must be positive");
    }

    py::array_t<vectorforge::idx_t> ids(
        {static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
    py::array_t<float> distances({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
    self.search_batch(ptr, nq, k, ids.mutable_data(), distances.mutable_data());
    if (squeeze) {
        return py::make_tuple(ids.reshape({static_cast<py::ssize_t>(k)}),
                              distances.reshape({static_cast<py::ssize_t>(k)}));
    }
    return py::make_tuple(ids, distances);
}

template <typename Index>
void add_numpy(Index& self, F32C vectors) {
    const auto buf = vectors.request();
    if (buf.ndim != 2) {
        throw py::value_error("vectors must have shape (n, dim)");
    }
    check_dim(self.dim(), buf.shape[1]);
    self.add(static_cast<const float*>(buf.ptr), static_cast<std::size_t>(buf.shape[0]));
}

}  // namespace

PYBIND11_MODULE(_vectorforge, m) {
    m.doc() = "VectorForge Python bindings (Phase 2: FlatIndex + HNSWIndex)";
    m.attr("__version__") = VECTORFORGE_VERSION;

    py::class_<vectorforge::FlatIndex>(m, "FlatIndex")
        .def(py::init<vectorforge::dim_t, const std::string&>(),
             py::arg("dim"),
             py::arg("metric") = "l2")
        .def_property_readonly("dim", &vectorforge::FlatIndex::dim)
        .def_property_readonly("metric",
                               [](const vectorforge::FlatIndex& self) {
                                   return std::string(vectorforge::metric_name(self.metric()));
                               })
        .def_property_readonly("ntotal", &vectorforge::FlatIndex::size)
        .def(
            "add",
            [](vectorforge::FlatIndex& self, F32C vectors) { add_numpy(self, vectors); },
            py::arg("vectors"))
        .def(
            "search",
            [](const vectorforge::FlatIndex& self, F32C queries, int k) {
                return search_numpy(self, queries, k);
            },
            py::arg("queries"),
            py::arg("k") = 10)
        .def("save", &vectorforge::FlatIndex::save, py::arg("path"))
        .def(
            "load",
            [](vectorforge::FlatIndex& self, const std::string& path) { self.load(path); },
            py::arg("path"));

    py::class_<vectorforge::HNSWIndex>(m, "HNSWIndex")
        .def(py::init<vectorforge::dim_t,
                      const std::string&,
                      std::size_t,
                      std::size_t,
                      std::size_t,
                      std::uint64_t>(),
             py::arg("dim"),
             py::arg("metric") = "l2",
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("ef_search") = 100,
             py::arg("seed") = 42)
        .def_property_readonly("dim", &vectorforge::HNSWIndex::dim)
        .def_property_readonly("metric",
                               [](const vectorforge::HNSWIndex& self) {
                                   return std::string(vectorforge::metric_name(self.metric()));
                               })
        .def_property_readonly("ntotal", &vectorforge::HNSWIndex::size)
        .def_property_readonly("M", &vectorforge::HNSWIndex::M)
        .def_property_readonly("ef_construction", &vectorforge::HNSWIndex::ef_construction)
        .def_property(
            "ef_search", &vectorforge::HNSWIndex::ef_search, &vectorforge::HNSWIndex::set_ef_search)
        .def_property_readonly("seed", &vectorforge::HNSWIndex::seed)
        .def_property_readonly("max_level", &vectorforge::HNSWIndex::max_level)
        .def_property_readonly("entry_point", &vectorforge::HNSWIndex::entry_point)
        .def("graph_digest", &vectorforge::HNSWIndex::graph_digest)
        .def(
            "add",
            [](vectorforge::HNSWIndex& self, F32C vectors) { add_numpy(self, vectors); },
            py::arg("vectors"))
        .def(
            "search",
            [](const vectorforge::HNSWIndex& self, F32C queries, int k) {
                return search_numpy(self, queries, k);
            },
            py::arg("queries"),
            py::arg("k") = 10)
        .def("save", &vectorforge::HNSWIndex::save, py::arg("path"))
        .def(
            "load",
            [](vectorforge::HNSWIndex& self, const std::string& path) { self.load(path); },
            py::arg("path"));
}
