# Contributing to snforacle

Thank you for your interest in contributing!  Bug reports, documentation fixes,
new backends, and test improvements are all welcome.

## Development setup

```bash
git clone https://github.com/antieau/snforacle
cd snforacle
pip install -e ".[dev,cypari2,flint]"
pytest tests/           # sage/magma tests auto-skip if binaries not on PATH
pytest --cov=snforacle tests/
```

## Running the benchmarks

```bash
python benchmarks/bench.py
```

Results are printed as ASCII tables and saved to `benchmarks/results.csv` and
`benchmarks/ff_results.csv`.

---

## Project conventions

- **Python 3.10+**; use `from __future__ import annotations` in every module.
- **Type hints** on all public and private functions.
- **NumPy-style docstrings** (Parameters / Returns sections) for non-trivial functions.
- `list[list[int]]` for internal dense integer matrix representation.
- `Poly = list[int]` (coefficient list, constant-term first, entries in `[0, p-1]`,
  `[]` = zero polynomial, no trailing zeros) for polynomial entries.
- Private helpers prefixed with `_`; backend classes named `<Name>Backend` /
  `<Name>PolyBackend` / `<Name>FFBackend`.
- CLI backends write scripts to `tempfile.TemporaryDirectory()`, run
  `subprocess.run([binary, ...], timeout=_TIMEOUT)`, and parse stdout.

---

## Adding an integer matrix backend

1. Create `snforacle/backends/<name>.py`.
2. Subclass `SNFBackend` from `snforacle.backends.base`:

   ```python
   from snforacle.backends.base import SNFBackend

   class MyBackend(SNFBackend):
       def compute_snf(self, matrix, nrows, ncols):
           ...
       def compute_snf_with_transforms(self, matrix, nrows, ncols):
           ...
       def compute_hnf(self, matrix, nrows, ncols):
           ...
       def compute_hnf_with_transform(self, matrix, nrows, ncols):
           ...
       def compute_elementary_divisors(self, matrix, nrows, ncols):
           ...
   ```

   Raise `NotImplementedError` for operations the backend cannot support.

3. Register the backend in `snforacle/interface.py` by adding a lazy-loader
   function in the `_register_*` section and including the backend name in
   `_OP_PRIORITY`.

4. Add tests in `tests/test_<name>.py` following the structure of
   `tests/test_pure_python.py`.

## Adding a finite-field (F_p) backend

Same pattern, but subclass `FFBackend` from `snforacle.backends.ff_base` and
implement:

```
compute_snf, compute_snf_with_transforms,
compute_hnf, compute_hnf_with_transform, compute_rank
```

Register in `snforacle/ff_interface.py`.  Add tests in `tests/test_<name>_ff.py`.

## Adding a polynomial (F_p[x]) backend

Subclass `PolyBackend` from `snforacle.backends.poly_base`.  Each method takes an
additional `p: int` argument:

```
compute_snf, compute_snf_with_transforms,
compute_hnf, compute_hnf_with_transform, compute_elementary_divisors
```

Register in `snforacle/poly_interface.py`.  Add tests in
`tests/test_<name>_poly.py`.

---

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your changes, add tests, and ensure `pytest tests/` passes.
3. Open a pull request with a clear description of the motivation and approach.

Please open an issue first for significant design changes so we can discuss
the approach before a large pull request is submitted.
