# snforacle

Smith normal form (SNF), Hermite normal form (HNF), and elementary divisors of integer matrices, **polynomial matrices over F_p[x]**, and **matrices over finite fields F_p** via a uniform JSON interface, with pluggable backends: PARI/GP (cypari2), FLINT (python-flint), SageMath (CLI subprocess), MAGMA (CLI subprocess), and a pure Python reference implementation.

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
├── __init__.py        # Public API: integer + poly_* + ff_* ops + all schemas
├── interface.py       # Integer matrix entry-points; dispatches to backends
├── poly_interface.py  # Poly matrix entry-points; dispatches to poly backends
├── ff_interface.py    # F_p matrix entry-points; dispatches to FF backends
├── schema.py          # Pydantic v2 input/output models (integer matrices)
├── poly_schema.py     # Pydantic v2 input/output models (polynomial matrices)
├── ff_schema.py       # Pydantic v2 input/output models (F_p matrices)
└── backends/
    ├── base.py              # Abstract SNFBackend (integer)
    ├── poly_base.py         # Abstract PolyBackend (F_p[x])
    ├── ff_base.py           # Abstract FFBackend (F_p)
    ├── cypari2.py           # PARI/GP backend (default for SNF/ED; no HNF)
    ├── flint.py             # FLINT backend (SNF+ED+HNF; no transforms)
    ├── sage.py              # SageMath CLI backend (all integer ops)
    ├── magma.py             # MAGMA CLI backend (all integer ops)
    ├── pure_python.py       # Pure Python integer reference implementation
    ├── sage_poly.py         # SageMath CLI backend (all poly ops)
    ├── magma_poly.py        # MAGMA CLI backend (all poly ops)
    ├── pure_python_poly.py  # Pure Python poly reference implementation
    ├── flint_ff.py          # FLINT backend (SNF+HNF+rank via nmod_mat)
    ├── sage_ff.py           # SageMath CLI backend (all F_p ops)
    ├── magma_ff.py          # MAGMA CLI backend (all F_p ops)
    └── pure_python_ff.py    # Pure Python F_p reference implementation
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
├── test_random_crossbackend.py # Random cross-backend consistency
├── test_pure_python_ff.py  # F_p pure_python backend tests + public API
├── test_flint_ff.py        # F_p flint backend tests (skipped if no flint)
├── test_sage_ff.py         # F_p sage tests (skipped if no sage)
├── test_magma_ff.py        # F_p MAGMA tests (skipped if no magma)
└── test_evil_ff.py         # Evil edge cases across all F_p backends
benchmarks/
├── __init__.py
└── bench.py           # Timing suite: integer + F_p variants, all backends
```

## Conventions

- **Python 3.10+**; use `from __future__ import annotations` for forward refs
- **Type hints** everywhere; Pydantic v2 models for all public I/O
- **NumPy-style docstrings** (Parameters / Returns sections)
- **Integer backends**: inherit `SNFBackend`; **Poly backends**: inherit `PolyBackend`; **FF backends**: inherit `FFBackend`
- CLI backends (sage, magma) write temp files to `tempfile.TemporaryDirectory()`, run `subprocess.run(..., timeout=120)`, and parse stdout
- Private helpers prefixed with `_`; backend classes named `<Name>Backend` / `<Name>PolyBackend` / `<Name>FFBackend`
- Internal integer ops use `list[list[int]]`; poly ops use `list[list[Poly]]` where `Poly = list[int]`; FF ops use `list[list[int]]` with entries in `[0, p-1]`
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

## Backend Capabilities — Finite Field Matrices over F_p

| Backend | SNF | SNF+T | HNF | HNF+T | Rank | How installed |
|---------|-----|-------|-----|-------|------|---------------|
| `flint` | yes | yes*  | yes | yes*  | yes  | `pip install snforacle[flint]` |
| `pure_python` | yes | yes | yes | yes | yes | stdlib only |
| `sage`  | yes | yes   | yes | yes   | yes  | SageMath on PATH |
| `magma` | yes | yes   | yes | yes   | yes  | MAGMA on PATH |

**Notes (finite field):**
- `flint` uses `nmod_mat.rref()` for all no-transform operations; transform variants delegate to `pure_python_ff`.
- Default priority for no-transform ops: flint → pure_python → sage → magma.
- Default priority for transform ops: pure_python → sage → magma.
- Sage `.sage` scripts: integer literals in `sum(1 for ...)` become Sage `Integer`; always wrap with `int()` before JSON serialisation.
- MAGMA: use `EchelonForm(M)` (functional, returns new matrix) not `EchelonForm(~M)` (in-place, unsupported for GF(p) matrices).
- MAGMA label lines must include `\n` (e.g. `printf "SNF\n"`) so the block parser sees the label alone on its own line.

## Optional Extras

| Extra | Dependency | Backend |
|-------|-----------|---------|
| `cypari2` | `cypari2>=2.2` | PARI/GP (default backend) |
| `flint` | `python-flint>=0.8.0` | FLINT |
| `dev` | `pytest`, `numpy` | — |

sage and magma are external CLI tools; they have no pip extras.
