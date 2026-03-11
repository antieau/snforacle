# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-03-10

Initial public release.

### Added

- **Integer matrices** — Smith normal form (SNF), SNF with unimodular transforms,
  Hermite normal form (HNF), HNF with left transform, and elementary divisors for
  matrices over ℤ.
- **Finite-field matrices** — the same five operations for matrices over F_p, plus
  a dedicated `ff_rank` function.  Over a field the SNF is always
  `diag(1,…,1,0,…,0)` and the HNF is the reduced row echelon form (RREF).
- **Polynomial matrices** — all five operations for matrices over F_p[x].
- **Backends**
  - `cypari2` — PARI/GP via `cypari2`; default for integer SNF and elementary divisors.
  - `flint` — FLINT via `python-flint`; default for integer HNF and F_p operations.
  - `sage` — SageMath CLI subprocess; supports all operations on all matrix types.
  - `magma` — MAGMA CLI subprocess; supports all operations on all matrix types.
  - `pure_python` — stdlib-only reference implementation; suitable for small matrices.
- **Uniform JSON/Pydantic API** — all functions accept plain `dict` or Pydantic
  model input; all results are Pydantic models with `.model_dump()` /
  `.model_dump_json()` serialisation.
- **Sparse input** — `SparseIntMatrix` / `SparseFFMatrix` / `SparsePolyMatrix` for
  sparse matrix input; unlisted entries are implicitly zero.
- Comprehensive test suite (530+ tests) with cross-backend consistency checks and
  adversarial edge-case coverage.
- Benchmarking suite (`benchmarks/bench.py`) timing all backends on dense/sparse
  integer matrices and dense finite-field matrices across multiple sizes.

### Known limitations

- `cypari2` does not support row HNF (PARI's `mathnf()` computes column HNF with
  an incompatible convention).
- `flint` does not support SNF/HNF with transform matrices (python-flint 0.8.0
  limitation).  For F_p transforms, `flint` delegates to the `pure_python` backend.
- `fmpz_mat_snf()` in FLINT 3.4.0 hangs on certain non-square integer matrices;
  the `flint` backend is therefore excluded from the default SNF/ED path for
  non-square inputs (see `FLINT_BUG.md`).
- The `pure_python` integer backend has O(n⁴) complexity with exponential
  intermediate-value growth; practical limit is roughly 12×12.
- There is no `flint` polynomial backend; python-flint 0.8.0 does not expose
  `nmod_poly_mat`.
