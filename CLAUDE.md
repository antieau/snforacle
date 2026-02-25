# snforacle

Smith normal form (SNF) of integer matrices via a uniform JSON interface, with pluggable backends: PARI/GP (cypari2), FLINT (python-flint), SageMath (CLI subprocess), and MAGMA (CLI subprocess).

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
│                      #   DenseIntMatrix, SparseIntMatrix, SNFResult, SNFWithTransformsResult
├── interface.py       # Entry-point functions; dispatches to backends
├── schema.py          # Pydantic v2 input/output models
└── backends/
    ├── base.py        # Abstract SNFBackend base class
    ├── cypari2.py     # PARI/GP backend (default)
    ├── flint.py       # FLINT backend (SNF only; transforms not supported)
    ├── sage.py        # SageMath CLI backend (requires `sage` on PATH)
    └── magma.py       # MAGMA CLI backend (requires `magma` on PATH)
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

| Backend | SNF | Transforms | How installed |
|---------|-----|------------|---------------|
| `cypari2` | yes | yes | `pip install snforacle[cypari2]` |
| `flint`   | yes | no  | `pip install snforacle[flint]` |
| `sage`    | yes | yes | SageMath installed separately; `sage` on PATH |
| `magma`   | yes | yes | MAGMA installed separately; `magma` on PATH |

MAGMA raises `ValueError` for matrices where `nrows * ncols > 10_000_000` (embedding that many integers inline in a script is impractical).

## Optional Extras

| Extra | Dependency | Backend |
|-------|-----------|---------|
| `cypari2` | `cypari2>=2.2` | PARI/GP (default backend) |
| `flint` | `python-flint>=0.8.0` | FLINT |
| `dev` | `pytest`, `numpy` | — |

sage and magma are external CLI tools; they have no pip extras.
