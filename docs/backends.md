# Backends

snforacle dispatches to pluggable backends. Every backend accepts the same input and returns the same output types. Unavailable backends raise a clear error when first used.

## Integer matrix backends

### cypari2

The default backend for SNF and elementary divisors. Wraps [PARI/GP](https://pari.math.u-bordeaux.fr/) via the `cypari2` Python binding.

- **Install:** `pip install "snforacle[cypari2]"`
- **SNF:** fast; uses PARI's `matsnf()`
- **Elementary divisors:** uses the same path; returns the invariant factors directly
- **HNF:** not supported — PARI's `mathnf()` computes the *column* HNF with an incompatible convention. Use `flint`, `sage`, or `magma` for row HNF.

### flint

Wraps [FLINT](https://flintlib.org/) via the `python-flint` binding.

- **Install:** `pip install "snforacle[flint]"`
- **SNF + elementary divisors:** supported; uses `fmpz_mat.snf()`
- **HNF:** supported (no transforms); uses `fmpz_mat.hnf()`
- **Transforms:** not supported (python-flint 0.8.0 does not expose transform tracking)

!!! warning "Known FLINT bug"
    `fmpz_mat.snf()` in FLINT 3.4.0 hangs indefinitely on certain non-square matrices. The `flint` backend is therefore excluded from the default backend priority for SNF and elementary divisors when a non-square matrix is detected. See `FLINT_BUG.md` in the repository for the full reproducer.

### sage

Wraps [SageMath](https://www.sagemath.org/) via a CLI subprocess. Supports all five operations.

- **Install:** install SageMath separately and ensure `sage` is on your `PATH`
- All operations are supported including transforms

### magma

Wraps [MAGMA](http://magma.maths.usyd.edu.au/) via a CLI subprocess. Supports all five operations.

- **Install:** install MAGMA separately and ensure `magma` is on your `PATH`
- All operations are supported including transforms

### pure_python

A pure Python reference implementation. No dependencies beyond the standard library.

- **Install:** always available
- Suitable for matrices up to roughly 12×12; uses O(n⁴) algorithm with exponential intermediate-value growth

## Finite-field matrix backends (F_p)

### flint (default for FF)

Uses FLINT's `nmod_mat` type for fast in-process computation over F_p.

- SNF, HNF, and rank without transforms: all fast
- Transforms: delegate to `pure_python` (python-flint 0.8.0 does not expose transform tracking from `nmod_mat`)

### pure_python (FF)

O(n³) Gaussian elimination in Python. Handles matrices up to roughly n ≈ 100.

### sage / magma (FF)

Same subprocess approach as for integer matrices. Use these for the fastest transforms or very large matrices.

## Polynomial matrix backends (F_p[x])

There is no `flint` polynomial backend — python-flint 0.8.0 does not expose `nmod_poly_mat`.

### sage (default for poly)

Uses SageMath's native polynomial matrix algorithms. Recommended for all non-trivial matrices.

### magma (poly)

Uses MAGMA's polynomial matrix algorithms.

### pure_python (poly)

GCD-based Euclidean SNF/HNF. Reference implementation; suitable for small matrices only (n ≲ 10).

## Default backend selection

Backends are selected automatically based on what is available:

| Operation | Priority order |
|-----------|---------------|
| Integer SNF | cypari2 → flint → magma → sage → pure_python |
| Integer SNF+transforms | cypari2 → magma → sage → pure_python |
| Integer HNF | flint → magma → sage → pure_python |
| Integer HNF+transform | magma → sage → pure_python |
| Integer elem. div. | cypari2 → flint → magma → sage → pure_python |
| FF SNF / HNF / rank | flint → pure_python → sage → magma |
| FF SNF+T / HNF+T | pure_python → sage → magma |
| Poly all operations | sage → magma → pure_python |

To override the default, pass `backend="name"` to any function:

```python
result = smith_normal_form(M, backend="sage")
result = ff_rank(M, backend="magma")
result = poly_smith_normal_form(M, backend="pure_python")
```
