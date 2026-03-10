# snforacle

[![CI](https://github.com/antieau/snforacle/actions/workflows/ci.yml/badge.svg)](https://github.com/antieau/snforacle/actions/workflows/ci.yml)

**snforacle** computes the [Smith normal form](https://en.wikipedia.org/wiki/Smith_normal_form) (SNF), [Hermite normal form](https://en.wikipedia.org/wiki/Hermite_normal_form) (HNF), and elementary divisors of integer matrices, polynomial matrices over F_p[x], and matrices over finite fields F_p — all through a single, uniform Python API, regardless of which backend does the actual computation.

The **Smith normal form** of an integer matrix M is the unique diagonal matrix D = diag(d₁, d₂, …, dᵣ) with d₁ | d₂ | … | dᵣ such that D = U · M · V for some invertible integer matrices U, V. The diagonal entries are the **invariant factors** of M; they appear in homology computations, lattice problems, and number theory.

The **Hermite Normal Form** (row HNF) is the unique upper-triangular matrix H with positive pivots (smallest to largest) satisfying H = U · M for some unimodular integer matrix U.

The **elementary divisors** are the non-zero diagonal entries of the SNF, returned in non-decreasing order. They can be computed via a potentially faster dedicated path.

## Backends

| Backend | Requires | SNF | SNF+T | HNF | HNF+T | Elem. Div. |
|---------|----------|-----|-------|-----|-------|-----------|
| `cypari2` *(default)* | `pip install snforacle[cypari2]` | yes | yes | no | no | yes |
| `flint` | `pip install snforacle[flint]` | yes | no | yes | no | yes |
| `sage` | SageMath on PATH | yes | yes | yes | yes | yes |
| `magma` | MAGMA on PATH | yes | yes | yes | yes | yes |
| `pure_python` | none (stdlib only) | yes | yes | yes | yes | yes |

All backends accept the same input and return the same output types. Backends that are unavailable raise a clear error when first used.

**Notes:**
- The `cypari2` backend does not support row HNF because PARI's native `mathnf()` computes the column HNF (H = M·U) which uses an incompatible convention. For HNF, use the `flint`, `sage`, or `magma` backend (default: `flint`).
- The `flint` backend does not support SNF or HNF with transforms (python-flint 0.8.0 limitation). Use `sage`, `magma`, or `pure_python` for transforms.
- The `pure_python` backend is an educational reference with O(n⁴) complexity and exponential intermediate-value growth. It is suitable for matrices up to roughly 12×12; use a faster backend for anything larger.

## Installation

```bash
# PARI/GP backend (recommended default)
pip install "snforacle[cypari2]"

# FLINT backend
pip install "snforacle[flint]"

# Both
pip install "snforacle[cypari2,flint]"

# SageMath and MAGMA backends need no pip package —
# install SageMath or MAGMA separately and make sure
# 'sage' / 'magma' is on your PATH.
```

## Quick start

```python
from snforacle import (
    smith_normal_form,
    smith_normal_form_with_transforms,
    hermite_normal_form,
    hermite_normal_form_with_transform,
    elementary_divisors,
)

# Dense input
M = {
    "format": "dense",
    "nrows": 3,
    "ncols": 3,
    "entries": [[2, 4, 4], [-6, 6, 12], [10, -4, -16]],
}

# Smith Normal Form
result = smith_normal_form(M)
print(result.invariant_factors)          # [2, 6, 12]
print(result.smith_normal_form.entries)  # [[2,0,0],[0,6,0],[0,0,12]]

# Smith Normal Form with unimodular transforms U, V such that U @ M @ V == D
result = smith_normal_form_with_transforms(M)
print(result.left_transform.entries)     # 3×3 integer matrix U
print(result.right_transform.entries)    # 3×3 integer matrix V

# Hermite Normal Form (row HNF: H = U·M)
result = hermite_normal_form(M, backend="flint")
print(result.hermite_normal_form.entries)  # [[2,4,4],[0,6,0],[0,0,12]]

# HNF with transform
result = hermite_normal_form_with_transform(M, backend="sage")
print(result.left_transform.entries)     # 3×3 unimodular matrix U

# Elementary divisors (potentially faster than SNF)
result = elementary_divisors(M)
print(result.elementary_divisors)        # [2, 6, 12]

# Choose a different backend
result = smith_normal_form(M, backend="flint")
result = smith_normal_form(M, backend="sage")   # requires sage on PATH
result = smith_normal_form(M, backend="magma")  # requires magma on PATH
```

## Input formats

Both functions accept a `DenseIntMatrix` or `SparseIntMatrix` (as a Pydantic model or a plain `dict`).

**Dense** — full list of rows:
```python
{
    "format": "dense",
    "nrows": 2,
    "ncols": 3,
    "entries": [[1, 2, 3], [4, 5, 6]],
}
```

**Sparse** — list of `(row, col, value)` triples; unlisted entries are zero:
```python
{
    "format": "sparse",
    "nrows": 3,
    "ncols": 3,
    "entries": [
        {"row": 0, "col": 0, "value": 2},
        {"row": 1, "col": 1, "value": 6},
        {"row": 2, "col": 2, "value": 12},
    ],
}
```

## Using Pydantic models directly

All public functions accept either a plain `dict` **or** an instantiated Pydantic model. You can also import the input and output models for type-safe construction and serialisation:

```python
from snforacle import smith_normal_form, elementary_divisors
from snforacle.schema import DenseIntMatrix, SparseIntMatrix, SparseEntry

# Build input from a Pydantic model instead of a plain dict
M = DenseIntMatrix(
    format="dense",
    nrows=3,
    ncols=3,
    entries=[[2, 4, 4], [-6, 6, 12], [10, -4, -16]],
)
result = smith_normal_form(M)
print(result.invariant_factors)          # [2, 6, 12]

# Sparse input via Pydantic
S = SparseIntMatrix(
    format="sparse",
    nrows=3,
    ncols=3,
    entries=[
        SparseEntry(row=0, col=0, value=2),
        SparseEntry(row=1, col=1, value=6),
        SparseEntry(row=2, col=2, value=12),
    ],
)
result = elementary_divisors(S)
print(result.elementary_divisors)        # [2, 6, 12]

# Serialise output to dict or JSON
print(result.model_dump())
print(result.model_dump_json())
```

The same pattern applies to finite-field and polynomial matrix inputs:

```python
from snforacle import ff_smith_normal_form, poly_smith_normal_form
from snforacle.ff_schema import DenseFFMatrix
from snforacle.poly_schema import DensePolyMatrix

M_ff = DenseFFMatrix(
    format="dense_ff",
    nrows=2,
    ncols=2,
    p=7,
    entries=[[1, 2], [3, 4]],
)
result = ff_smith_normal_form(M_ff)
print(result.rank)                       # 2

M_poly = DensePolyMatrix(
    format="dense_poly",
    nrows=2,
    ncols=2,
    p=2,
    entries=[[[0, 1], [1]], [[], [1, 1]]],  # [[x, 1], [0, x+1]] over F_2[x]
)
result = poly_smith_normal_form(M_poly)
print(result.invariant_factors)
```

## Output

**Smith Normal Form:**

`smith_normal_form(M)` returns an `SNFResult` with:
- `smith_normal_form` — the m×n SNF matrix as a `DenseIntMatrix`
- `invariant_factors` — list of nonzero diagonal entries `[d₁, …, dᵣ]`

`smith_normal_form_with_transforms(M)` returns an `SNFWithTransformsResult` with the above plus:
- `left_transform` — unimodular m×m matrix U
- `right_transform` — unimodular n×n matrix V
- Satisfies: U @ M @ V = SNF

**Hermite Normal Form:**

`hermite_normal_form(M)` returns an `HNFResult` with:
- `hermite_normal_form` — the m×n HNF matrix as a `DenseIntMatrix`

`hermite_normal_form_with_transform(M)` returns an `HNFWithTransformResult` with:
- `hermite_normal_form` — the m×n HNF matrix
- `left_transform` — unimodular m×m matrix U
- Satisfies: U @ M = HNF

**Elementary Divisors:**

`elementary_divisors(M)` returns an `ElementaryDivisorsResult` with:
- `elementary_divisors` — list of nonzero invariant factors in non-decreasing order

All output models have `.model_dump()` and `.model_dump_json()` for serialisation.

## Matrices over finite fields F_p

snforacle supports matrices whose entries are elements of F_p (integers mod a prime p). Over a field every nonzero element is a unit, so the SNF is always `diag(1,…,1,0,…,0)` and the HNF is the reduced row echelon form (RREF).

```python
from snforacle import (
    ff_smith_normal_form,
    ff_smith_normal_form_with_transforms,
    ff_hermite_normal_form,
    ff_hermite_normal_form_with_transform,
    ff_rank,
)

# Dense input over F_7
M = {
    "format": "dense_ff",
    "nrows": 3,
    "ncols": 3,
    "p": 7,
    "entries": [[1, 2, 3], [4, 5, 6], [0, 1, 2]],
}

result = ff_smith_normal_form(M)
print(result.rank)                           # 2
print(result.smith_normal_form.entries)      # [[1,0,0],[0,1,0],[0,0,0]]

result = ff_smith_normal_form_with_transforms(M)
# result.left_transform @ M @ result.right_transform == smith_normal_form

result = ff_hermite_normal_form(M)
print(result.hermite_normal_form.entries)    # RREF of M

result = ff_hermite_normal_form_with_transform(M)
# result.left_transform @ M == hermite_normal_form

result = ff_rank(M)
print(result.rank)                           # 2

# Sparse input
M_sparse = {
    "format": "sparse_ff",
    "nrows": 3,
    "ncols": 3,
    "p": 7,
    "entries": [
        {"row": 0, "col": 0, "value": 1},
        {"row": 1, "col": 1, "value": 1},
    ],
}
result = ff_rank(M_sparse)
print(result.rank)   # 2
```

**Finite field backend availability:**

| Backend | Requires | SNF | SNF+T | HNF | HNF+T | Rank |
|---------|----------|-----|-------|-----|-------|------|
| `flint` *(default)* | `pip install snforacle[flint]` | yes | yes* | yes | yes* | yes |
| `pure_python` | none (stdlib only) | yes | yes | yes | yes | yes |
| `sage` | SageMath on PATH | yes | yes | yes | yes | yes |
| `magma` | MAGMA on PATH | yes | yes | yes | yes | yes |

**Notes:**
- For `snf` and `hnf` without transforms the default is `flint` (fast in-process RREF via `nmod_mat`); falls back to `pure_python` if flint is not installed.
- For operations with transforms the default is `pure_python` (no subprocess overhead); `sage` or `magma` can be selected explicitly.
- *`flint` transform operations delegate to the pure Python algorithm (python-flint 0.8.0 does not expose transform tracking from `nmod_mat`).
- `pure_python` is O(n³) in field operations but Python-speed; it handles matrices up to roughly n ≈ 100 comfortably.

## Polynomial matrices over F_p[x]

snforacle also handles matrices whose entries are polynomials over a finite field F_p (p prime). All five operations are supported.

```python
from snforacle import (
    poly_smith_normal_form,
    poly_smith_normal_form_with_transforms,
    poly_hermite_normal_form,
    poly_hermite_normal_form_with_transform,
    poly_elementary_divisors,
)

# Polynomials are coefficient lists [c_0, c_1, ..., c_d] (constant term first),
# with coefficients in [0, p-1]. The zero polynomial is [].
# Here: [[x, 1], [0, x+1]] over F_2[x]
M = {
    "format": "dense_poly",
    "nrows": 2,
    "ncols": 2,
    "p": 2,
    "entries": [[[0, 1], [1]], [[], [1, 1]]],
}

result = poly_smith_normal_form(M)
print(result.invariant_factors)   # monic polynomials, e.g. [[1], [0, 0, 1]]

result = poly_smith_normal_form_with_transforms(M)
# result.left_transform @ M @ result.right_transform == result.smith_normal_form

result = poly_hermite_normal_form(M)
result = poly_elementary_divisors(M)
```

**Polynomial backend availability:**

| Backend | Requires | SNF | SNF+T | HNF | HNF+T | Elem. Div. |
|---------|----------|-----|-------|-----|-------|-----------|
| `sage` *(default)* | SageMath on PATH | yes | yes | yes | yes | yes |
| `magma` | MAGMA on PATH | yes | yes | yes | yes | yes |
| `pure_python` | none (stdlib only) | yes | yes | yes | yes | yes |

**Notes:**
- Invariant factors and HNF pivots are always returned as monic polynomials.
- `pure_python` is a reference implementation suitable for small matrices only (n ≲ 10).
- There is no `flint` polynomial backend; python-flint 0.8.0 does not expose `nmod_poly_mat`.

## Benchmarks

```bash
pip install ".[dev,cypari2,flint]"
python benchmarks/bench.py
```

This times all available backends on random matrices at multiple sizes. Integer results are printed as an ASCII table and saved to `benchmarks/results.csv`; finite-field results are saved to `benchmarks/ff_results.csv`.

## Development

```bash
pip install ".[dev,cypari2,flint]"
pytest tests/           # sage/magma tests auto-skip if not on PATH
pytest --cov=snforacle tests/
```

**Adding an integer matrix backend:** subclass `SNFBackend` in `snforacle/backends/` and implement all five abstract methods: `compute_snf`, `compute_snf_with_transforms`, `compute_hnf`, `compute_hnf_with_transform`, and `compute_elementary_divisors`. Register the class in `snforacle/interface.py`. Raise `NotImplementedError` for any operations the backend cannot support.

**Adding a polynomial matrix backend:** subclass `PolyBackend` in `snforacle/backends/` (same five methods, each taking an additional `p: int` argument) and register it in `snforacle/poly_interface.py`.

**Adding a finite-field matrix backend:** subclass `FFBackend` in `snforacle/backends/` and implement five methods: `compute_snf`, `compute_snf_with_transforms`, `compute_hnf`, `compute_hnf_with_transform`, and `compute_rank` — each taking `p: int`. Register the class in `snforacle/ff_interface.py`.
