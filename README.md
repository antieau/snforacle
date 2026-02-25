# snforacle

**snforacle** computes the [Smith normal form](https://en.wikipedia.org/wiki/Smith_normal_form) (SNF) of integer matrices through a single, uniform Python API — regardless of which backend does the actual computation.

The Smith normal form of an integer matrix M is the unique diagonal matrix D = diag(d₁, d₂, …, dᵣ) with d₁ | d₂ | … | dᵣ such that D = U · M · V for some invertible integer matrices U, V. The diagonal entries are the **invariant factors** of M; they appear in homology computations, lattice problems, and number theory.

## Backends

| Backend | Requires | SNF | Transforms |
|---------|----------|-----|------------|
| `cypari2` *(default)* | `pip install snforacle[cypari2]` | yes | yes |
| `flint` | `pip install snforacle[flint]` | yes | no |
| `sage` | SageMath on PATH | yes | yes |
| `magma` | MAGMA on PATH | yes | yes |

All backends accept the same input and return the same output types. Backends that are unavailable raise a clear error when first used.

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
from snforacle import smith_normal_form, smith_normal_form_with_transforms

# Dense input
M = {
    "format": "dense",
    "nrows": 3,
    "ncols": 3,
    "entries": [[2, 4, 4], [-6, 6, 12], [10, -4, -16]],
}

result = smith_normal_form(M)
print(result.invariant_factors)          # [2, 6, 12]
print(result.smith_normal_form.entries)  # [[2,0,0],[0,6,0],[0,0,12]]

# With unimodular transforms U, V such that U @ M @ V == D
result = smith_normal_form_with_transforms(M)
print(result.left_transform.entries)    # 3×3 integer matrix U
print(result.right_transform.entries)  # 3×3 integer matrix V

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

`smith_normal_form` returns an `SNFResult` with:
- `smith_normal_form` — the m×n SNF matrix as a `DenseIntMatrix`
- `invariant_factors` — list of nonzero diagonal entries `[d₁, …, dᵣ]`

`smith_normal_form_with_transforms` returns an `SNFWithTransformsResult` with the above plus:
- `left_transform` — unimodular m×m matrix U
- `right_transform` — unimodular n×n matrix V

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
