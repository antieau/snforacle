# Quick Start

## Integer matrices

```python
from snforacle import (
    smith_normal_form,
    smith_normal_form_with_transforms,
    hermite_normal_form,
    hermite_normal_form_with_transform,
    elementary_divisors,
)

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
result = hermite_normal_form(M)
print(result.hermite_normal_form.entries)  # [[2,4,4],[0,6,0],[0,0,12]]

# HNF with unimodular left transform
result = hermite_normal_form_with_transform(M)
print(result.left_transform.entries)     # 3×3 unimodular matrix U

# Elementary divisors (non-zero invariant factors)
result = elementary_divisors(M)
print(result.elementary_divisors)        # [2, 6, 12]

# Choose a specific backend
result = smith_normal_form(M, backend="flint")
result = smith_normal_form(M, backend="sage")   # requires sage on PATH
result = smith_normal_form(M, backend="magma")  # requires magma on PATH
```

## Using Pydantic models directly

All public functions accept either a plain `dict` **or** an instantiated Pydantic model:

```python
from snforacle import smith_normal_form, elementary_divisors
from snforacle.schema import DenseIntMatrix, SparseIntMatrix, SparseEntry

# Build input from a Pydantic model
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

## Finite-field matrices (F_p)

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

result = ff_hermite_normal_form(M)
print(result.hermite_normal_form.entries)    # RREF of M

result = ff_rank(M)
print(result.rank)                           # 2
```

## Polynomial matrices (F_p[x])

Polynomials are represented as coefficient lists `[c_0, c_1, ..., c_d]` (constant term first), with coefficients in `[0, p-1]`. The zero polynomial is `[]`.

```python
from snforacle import (
    poly_smith_normal_form,
    poly_smith_normal_form_with_transforms,
    poly_hermite_normal_form,
    poly_hermite_normal_form_with_transform,
    poly_elementary_divisors,
)

# [[x, 1], [0, x+1]] over F_2[x]
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
