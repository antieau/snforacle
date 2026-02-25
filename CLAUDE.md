# snforacle

Smith normal form (SNF), Hermite normal form (HNF), and elementary divisors of integer matrices via a uniform JSON interface, with pluggable backends: PARI/GP (cypari2), FLINT (python-flint), SageMath (CLI subprocess), and MAGMA (CLI subprocess).

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
├── __init__.py        # Public API: smith_normal_form, smith_normal_form_with_transforms,
│                      #   hermite_normal_form, hermite_normal_form_with_transform,
│                      #   elementary_divisors; plus result models and input schemas
├── interface.py       # Entry-point functions; dispatches to backends
├── schema.py          # Pydantic v2 input/output models
└── backends/
    ├── base.py        # Abstract SNFBackend base class
    ├── cypari2.py     # PARI/GP backend (default for SNF/ED; no HNF)
    ├── flint.py       # FLINT backend (SNF+ED+HNF; no transforms)
    ├── sage.py        # SageMath CLI backend (all operations)
    └── magma.py       # MAGMA CLI backend (all operations)
tests/
├── test_cypari2.py    # Schema validation, SNF correctness, transform tests
├── test_flint.py      # Flint tests + cross-backend consistency
├── test_sage.py       # Sage tests + cross-backend consistency (skipped if no sage)
└── test_magma.py      # MAGMA tests + cross-backend consistency (skipped if no magma)
benchmarks/
├── __init__.py
└── bench.py           # Timing suite: dense+sparse, 6 sizes, all backends
```

## Conventions

- **Python 3.10+**; use `from __future__ import annotations` for forward refs
- **Type hints** everywhere; Pydantic v2 models for all public I/O
- **NumPy-style docstrings** (Parameters / Returns sections)
- **Backends**: inherit `SNFBackend`, lazy-import the native library (or check CLI binary at construction), raise `NotImplementedError` for unsupported operations
- CLI backends (sage, magma) write temp files to `tempfile.TemporaryDirectory()`, run `subprocess.run(..., timeout=120)`, and parse stdout
- Private helpers prefixed with `_`; backend classes named `<Name>Backend`
- All internal matrix operations use plain `list[list[int]]`

## Backend Capabilities

| Backend | SNF | SNF+T | HNF | HNF+T | Elem. Div. | How installed |
|---------|-----|-------|-----|-------|-----------|---------------|
| `cypari2` | yes | yes   | no  | no    | yes       | `pip install snforacle[cypari2]` |
| `flint`   | yes | no    | yes | no    | yes       | `pip install snforacle[flint]` |
| `sage`    | yes | yes   | yes | yes   | yes       | SageMath on PATH |
| `magma`   | yes | yes   | yes | yes   | yes       | MAGMA on PATH |

**Notes:**
- `cypari2` does not support row HNF (PARI's `mathnf()` computes the column HNF with incompatible convention).
- `flint` does not support SNF/HNF with transforms (python-flint 0.8.0 limitation).
- `magma` raises `ValueError` for matrices where `nrows * ncols > 10_000_000` (inline script embedding limitation).

## Optional Extras

| Extra | Dependency | Backend |
|-------|-----------|---------|
| `cypari2` | `cypari2>=2.2` | PARI/GP (default backend) |
| `flint` | `python-flint>=0.8.0` | FLINT |
| `dev` | `pytest`, `numpy` | — |

sage and magma are external CLI tools; they have no pip extras.
