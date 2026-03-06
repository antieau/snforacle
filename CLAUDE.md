# snforacle

Smith normal form (SNF), Hermite normal form (HNF), and elementary divisors of integer matrices **and polynomial matrices over F_p[x]** via a uniform JSON interface, with pluggable backends: PARI/GP (cypari2), FLINT (python-flint), SageMath (CLI subprocess), MAGMA (CLI subprocess), and a pure Python reference implementation.

## Common Commands

```bash
# Install with all pip-installable backends and dev tools
pip install ".[dev,cypari2,flint]"

# Run tests (sage/magma tests auto-skip if binaries not on PATH)
pytest tests/

# Run tests with coverage
pytest --cov=snforacle tests/

# Run benchmarks (prints ASCII table, saves benchmarks/results.csv)
python benchmarks/bench.py
```

## Architecture

```
snforacle/
├── __init__.py        # Public API: integer ops + poly_* ops + all schemas
├── interface.py       # Integer matrix entry-points; dispatches to backends
├── poly_interface.py  # Poly matrix entry-points; dispatches to poly backends
├── schema.py          # Pydantic v2 input/output models (integer matrices)
├── poly_schema.py     # Pydantic v2 input/output models (polynomial matrices)
└── backends/
    ├── base.py              # Abstract SNFBackend (integer)
    ├── poly_base.py         # Abstract PolyBackend (F_p[x])
    ├── cypari2.py           # PARI/GP backend (default for SNF/ED; no HNF)
    ├── flint.py             # FLINT backend (SNF+ED+HNF; no transforms)
    ├── sage.py              # SageMath CLI backend (all integer ops)
    ├── magma.py             # MAGMA CLI backend (all integer ops)
    ├── pure_python.py       # Pure Python integer reference implementation
    ├── sage_poly.py         # SageMath CLI backend (all poly ops)
    ├── magma_poly.py        # MAGMA CLI backend (all poly ops)
    └── pure_python_poly.py  # Pure Python poly reference implementation
tests/
├── test_cypari2.py         # Schema validation, SNF correctness, transform tests
├── test_flint.py           # Flint tests + cross-backend consistency
├── test_sage.py            # Sage tests (skipped if no sage)
├── test_magma.py           # MAGMA tests (skipped if no magma)
├── test_pure_python.py     # Pure Python backend tests + cross-backend consistency
├── test_pure_python_poly.py # Poly pure_python backend tests
├── test_sage_poly.py       # Poly sage tests (skipped if no sage)
├── test_magma_poly.py      # Poly MAGMA tests (skipped if no magma)
├── test_evil_poly.py       # Evil edge cases across all poly backends
├── test_evil4.py           # Evil edge cases across all integer backends
└── test_random_crossbackend.py # Random cross-backend consistency
benchmarks/
├── __init__.py
└── bench.py           # Timing suite: dense+sparse, 6 sizes, all backends
```

## Conventions

- **Python 3.10+**; use `from __future__ import annotations` for forward refs
- **Type hints** everywhere; Pydantic v2 models for all public I/O
- **NumPy-style docstrings** (Parameters / Returns sections)
- **Integer backends**: inherit `SNFBackend`; **Poly backends**: inherit `PolyBackend`
- CLI backends (sage, magma) write temp files to `tempfile.TemporaryDirectory()`, run `subprocess.run(..., timeout=120)`, and parse stdout
- Private helpers prefixed with `_`; backend classes named `<Name>Backend` / `<Name>PolyBackend`
- Internal integer ops use `list[list[int]]`; poly ops use `list[list[Poly]]` where `Poly = list[int]`
- Polynomials: `[c_0, c_1, ..., c_d]` (constant-term first), entries in `[0, p-1]`, `[]` = zero poly, no trailing zeros

## Backend Capabilities — Integer Matrices

| Backend | SNF | SNF+T | HNF | HNF+T | Elem. Div. | How installed |
|---------|-----|-------|-----|-------|-----------|---------------|
| `cypari2` | yes | yes   | no  | no    | yes       | `pip install snforacle[cypari2]` |
| `flint`   | yes | no    | yes | no    | yes       | `pip install snforacle[flint]` |
| `sage`    | yes | yes   | yes | yes   | yes       | SageMath on PATH |
| `magma`   | yes | yes   | yes | yes   | yes       | MAGMA on PATH |
| `pure_python` | yes | yes | yes | yes | yes | stdlib only |

## Backend Capabilities — Polynomial Matrices over F_p[x]

| Backend | SNF | SNF+T | HNF | HNF+T | Elem. Div. | How installed |
|---------|-----|-------|-----|-------|-----------|---------------|
| `sage`    | yes | yes   | yes | yes   | yes       | SageMath on PATH |
| `magma`   | yes | yes   | yes | yes   | yes       | MAGMA on PATH |
| `pure_python` | yes | yes | yes | yes | yes | stdlib only |

**Notes (integer):**
- `cypari2` does not support row HNF (PARI's `mathnf()` computes the column HNF with incompatible convention).
- `flint` does not support SNF/HNF with transforms (python-flint 0.8.0 limitation).
- `flint.fmpz_mat.snf()` hangs on certain non-square matrices (FLINT 3.4.0 bug; see FLINT_BUG.md).
- `pure_python` is an educational reference with O(n⁴) complexity; suitable for small matrices only.

**Notes (polynomial):**
- python-flint 0.8.0 has no `nmod_poly_mat` type, so there is no flint poly backend.
- `pure_python` poly backend uses direct scalar elimination for unit pivots; otherwise GCD-based Euclidean SNF/HNF.
- Default backend priority: sage → magma → pure_python.

## Optional Extras

| Extra | Dependency | Backend |
|-------|-----------|---------|
| `cypari2` | `cypari2>=2.2` | PARI/GP (default backend) |
| `flint` | `python-flint>=0.8.0` | FLINT |
| `dev` | `pytest`, `numpy` | — |

sage and magma are external CLI tools; they have no pip extras.
