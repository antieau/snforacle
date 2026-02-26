# snforacle

**snforacle** computes the [Smith normal form](https://en.wikipedia.org/wiki/Smith_normal_form) (SNF), [Hermite normal form](https://en.wikipedia.org/wiki/Hermite_normal_form) (HNF), and elementary divisors of integer matrices through a single, uniform Python API — regardless of which backend does the actual computation.

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

**Note:** The `cypari2` backend does not support row HNF because PARI's native `mathnf()` computes the column HNF (H = M·U) which uses an incompatible convention. For HNF, use the `flint`, `sage`, or `magma` backend (default: `flint`).

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

## Benchmarks

```bash
pip install ".[dev,cypari2,flint]"
python benchmarks/bench.py
```

This times all available backends on dense and sparse random matrices at sizes 10, 50, 100, 500, 1 000, and 10 000. Results are printed as an ASCII table and saved to `benchmarks/results.csv`.

## Development

```bash
pip install ".[dev,cypari2,flint]"
pytest tests/           # sage/magma tests auto-skip if not on PATH
pytest --cov=snforacle tests/
```

To add a new backend, subclass `SNFBackend` in `snforacle/backends/`, implement `compute_snf` and `compute_snf_with_transforms`, then register it in `snforacle/interface.py`.
