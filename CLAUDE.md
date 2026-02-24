# snforacle

Smith normal form (SNF) of integer matrices via a uniform JSON interface, with pluggable backends (PARI/GP via cypari2, FLINT via python-flint).

## Common Commands

```bash
# Install with all backends and dev tools
pip install ".[dev,cypari2,flint]"

# Run tests
pytest tests/

# Run tests with coverage
pytest --cov=snforacle tests/
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
    └── flint.py       # FLINT backend
tests/
├── test_cypari2.py    # Schema validation, SNF correctness, transform tests
└── test_flint.py      # Flint tests + cross-backend consistency
```

## Conventions

- **Python 3.10+**; use `from __future__ import annotations` for forward refs
- **Type hints** everywhere; Pydantic v2 models for all public I/O
- **NumPy-style docstrings** (Parameters / Returns sections)
- **Backends**: inherit `SNFBackend`, lazy-import the native library, raise `NotImplementedError` for unsupported operations
- Private helpers prefixed with `_`; backend classes named `<Name>Backend`
- All internal matrix operations use plain `list[list[int]]`

## Optional Extras

| Extra | Dependency | Backend |
|-------|-----------|---------|
| `cypari2` | `cypari2>=2.2` | PARI/GP (default backend) |
| `flint` | `python-flint>=0.8.0` | FLINT |
| `dev` | `pytest` | — |
