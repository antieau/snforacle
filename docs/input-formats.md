# Input Formats

All public functions accept a Pydantic model instance **or** a plain `dict` that will be validated against the appropriate schema.

## Integer matrices

### Dense

Provide all entries as a list of rows:

```python
{
    "format": "dense",
    "nrows": 2,
    "ncols": 3,
    "entries": [[1, 2, 3], [4, 5, 6]],
}
```

Or using the Pydantic model:

```python
from snforacle.schema import DenseIntMatrix

M = DenseIntMatrix(format="dense", nrows=2, ncols=3, entries=[[1,2,3],[4,5,6]])
```

### Sparse

Provide only the non-zero entries as a list of `(row, col, value)` triples; all omitted positions are zero:

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

Or using Pydantic models:

```python
from snforacle.schema import SparseIntMatrix, SparseEntry

M = SparseIntMatrix(
    format="sparse",
    nrows=3,
    ncols=3,
    entries=[
        SparseEntry(row=0, col=0, value=2),
        SparseEntry(row=1, col=1, value=6),
        SparseEntry(row=2, col=2, value=12),
    ],
)
```

## Finite-field matrices (F_p)

The field characteristic `p` must be a prime. All entry values must be in `[0, p-1]`.

### Dense

```python
{
    "format": "dense_ff",
    "nrows": 2,
    "ncols": 2,
    "p": 7,
    "entries": [[1, 2], [3, 4]],
}
```

### Sparse

```python
{
    "format": "sparse_ff",
    "nrows": 3,
    "ncols": 3,
    "p": 7,
    "entries": [
        {"row": 0, "col": 0, "value": 1},
        {"row": 1, "col": 1, "value": 3},
    ],
}
```

## Polynomial matrices (F_p[x])

Each polynomial is a coefficient list `[c_0, c_1, ..., c_d]` with constant term first and coefficients in `[0, p-1]`. The zero polynomial is `[]`. No trailing zeros are permitted.

### Dense

```python
{
    "format": "dense_poly",
    "nrows": 2,
    "ncols": 2,
    "p": 2,
    # [[x, 1], [0, x+1]] over F_2[x]
    "entries": [[[0, 1], [1]], [[], [1, 1]]],
}
```

### Sparse

```python
{
    "format": "sparse_poly",
    "nrows": 3,
    "ncols": 3,
    "p": 5,
    "entries": [
        {"row": 0, "col": 0, "value": [1, 2]},   # 1 + 2x
        {"row": 1, "col": 1, "value": [0, 0, 1]}, # x^2
    ],
}
```

## Output models

All output models have `.model_dump()` and `.model_dump_json()` for serialisation.

| Function | Return type | Key fields |
|----------|-------------|-----------|
| `smith_normal_form` | `SNFResult` | `smith_normal_form`, `invariant_factors` |
| `smith_normal_form_with_transforms` | `SNFWithTransformsResult` | + `left_transform`, `right_transform` |
| `hermite_normal_form` | `HNFResult` | `hermite_normal_form` |
| `hermite_normal_form_with_transform` | `HNFWithTransformResult` | + `left_transform` |
| `elementary_divisors` | `ElementaryDivisorsResult` | `elementary_divisors` |
| `ff_smith_normal_form` | `FFSNFResult` | `smith_normal_form`, `rank` |
| `ff_hermite_normal_form` | `FFHNFResult` | `hermite_normal_form` |
| `ff_rank` | `FFRankResult` | `rank` |
| `poly_smith_normal_form` | `PolySNFResult` | `smith_normal_form`, `invariant_factors` |
| `poly_hermite_normal_form` | `PolyHNFResult` | `hermite_normal_form` |
| `poly_elementary_divisors` | `PolyElementaryDivisorsResult` | `elementary_divisors` |
