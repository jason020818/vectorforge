# Building VectorForge

## Requirements

* Linux x86-64 (the v0.1 target)
* C++20 compiler (GCC 11+ or Clang 14+)
* CMake 3.24+
* Python 3.10+
* Git (Catch2 is fetched on first C++ test configure)

Ninja is optional but faster:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

## C++ library and tests

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DVECTORFORGE_ENABLE_SANITIZERS=ON
cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CMake options

| option | default | meaning |
|--------|---------|---------|
| `VECTORFORGE_BUILD_TESTS` | ON | Fetch Catch2 and build `vectorforge_tests` |
| `VECTORFORGE_BUILD_PYTHON` | OFF | Build the pybind11 module (pip sets this ON) |
| `VECTORFORGE_ENABLE_SANITIZERS` | OFF | `-fsanitize=address,undefined` |

## Python package

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python -c "from vectorforge import FlatIndex; print(FlatIndex)"
pytest
ruff check python tests/python eval benchmarks
python eval/correctness.py --metric cosine
python eval/recall.py
```

`pip install` needs a working CMake and C++ compiler on PATH.

## Formatting

C++:

```bash
clang-format -i include/vectorforge/*.hpp src/**/*.cpp python/bindings.cpp tests/*.cpp
```

Python:

```bash
ruff format python tests/python eval benchmarks
```

## Clean clone check

From a fresh clone, the commands in the README C++ and Python sections must succeed without extra undocumented steps.

Before substantial changes, agents should read `docs/MASTER_SPEC.md` and `AGENTS.md` in addition to the build and architecture docs.

## GitHub Actions

The workflow definition is `ci/github-actions.yml`. It is also the file that belongs at `.github/workflows/ci.yml` so GitHub will run it. Enabling that path requires a Git credential with the `workflow` scope (`gh auth refresh -h github.com -s workflow`).
