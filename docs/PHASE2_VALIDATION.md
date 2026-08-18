# Phase 2 Synthetic Correctness / Recall Validation

Phase 2 synthetic correctness/recall validation.
Not a competitor performance benchmark.

## Environment

```text
commit:   8f1760ff3757da782b457f808b60cc66608637cd (base) + verification hardening
date:     2026-08-18
OS:       Linux 6.18.33.2-microsoft-standard-WSL2 x86_64
compiler: g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Python:   3.12.3
CPU:      Intel Core i7-10700 @ 2.90GHz
```

## HNSW configuration

```text
N:              10000
dim:            64
nq:             100
M:              16
efConstruction: 200
efSearch:       100
seed:           42
```

## Recall results

```text
L2 Recall@10:      0.941000
L2 Recall@100:     0.863500

Cosine Recall@10:  0.918000
Cosine Recall@100: 0.816200
```

Both metrics satisfy the Phase 2 gate: Recall@10 >= 0.90.

## Commands executed

```bash
# C++ Release
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DVECTORFORGE_BUILD_TESTS=ON
cmake --build build-release -j
ctest --test-dir build-release --output-on-failure
# 75/75 passed

# C++ Debug + ASan/UBSan
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug \
  -DVECTORFORGE_BUILD_TESTS=ON -DVECTORFORGE_ENABLE_SANITIZERS=ON
cmake --build build-debug -j
ctest --test-dir build-debug --output-on-failure
# 75/75 passed

# Python
pip install -e ".[dev]"
pytest                                     # 31/31 passed
python eval/correctness.py --metric l2     # PASS
python eval/correctness.py --metric cosine # PASS
python eval/recall.py --metric l2          # Recall@10 0.941000, PASS
python eval/recall.py --metric cosine      # Recall@10 0.918000, PASS
ruff check python tests/python eval benchmarks  # All checks passed
git ls-files '*.cpp' '*.hpp' | xargs clang-format --dry-run -Werror  # No errors
```

## Test summary

| gate                              | result |
|-----------------------------------|--------|
| C++ Release (75 tests)            | PASS   |
| C++ Debug + ASan/UBSan (75 tests) | PASS   |
| Python tests (31 tests)           | PASS   |
| FlatIndex regression              | PASS   |
| L2 Recall@10 >= 0.90              | PASS   |
| Cosine Recall@10 >= 0.90          | PASS   |
| Save/load continuation determinism| PASS   |
| VH01 NaN / +Inf / -Inf rejection  | PASS   |
| Ruff                              | PASS   |
| clang-format                      | PASS   |
