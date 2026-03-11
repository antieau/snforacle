# Maintainer notes

This document covers release management, CI, and documentation deployment.
For contributor-facing conventions see [CONTRIBUTING.md](CONTRIBUTING.md).

## Release process

1. **Update the version** in `pyproject.toml`:
   ```
   version = "0.2.0"
   ```

2. **Update `CHANGELOG.md`** — add a dated section for the new version.

3. **Commit and push to `main`**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "Release v0.2.0"
   git push
   ```
   CI runs the test suite; confirm it passes before proceeding.

4. **Publish a GitHub release:**
   - Go to **github.com/antieau/snforacle → Releases → Draft a new release**
   - Create a new tag matching the version: `v0.2.0`
   - Title: `v0.2.0`
   - Copy the relevant section from `CHANGELOG.md` as the release description
   - Click **Publish release**

5. **CI deploys the docs automatically** — publishing the release triggers the
   `docs` job, which runs `mkdocs gh-deploy --force` and pushes the rendered
   site to the `gh-pages` branch. No manual action required.

6. **Publish to PyPI** (when ready):
   ```bash
   pip install build twine
   python -m build
   twine upload dist/*
   ```

## CI overview

The workflow file is `.github/workflows/ci.yml`.

| Trigger | Jobs that run |
|---------|--------------|
| Push to `main` | `test` |
| Pull request to `main` | `test` |
| GitHub release published | `test` → `docs` (docs only deploy if tests pass) |

The `docs` job is intentionally gated on `test` via `needs: test`, so a broken
release can never publish documentation.

## Documentation

Docs are built with [MkDocs](https://www.mkdocs.org/) + the
[Material theme](https://squidfunk.github.io/mkdocs-material/).

**Source files** (committed to the repo):
- `mkdocs.yml` — site configuration and navigation
- `docs/` — Markdown source pages
- `docs/gen_schemas.py` — auto-generates the JSON Schemas page at build time
  from the live Pydantic models via `mkdocs-gen-files`

**Generated output** (never committed):
- `site/` — rendered HTML; excluded by `.gitignore`
- `gh-pages` branch — managed entirely by `mkdocs gh-deploy`; do not edit by hand

**To preview locally:**
```bash
pip install ".[docs,cypari2,flint]"
mkdocs serve          # live-reloading preview at http://127.0.0.1:8000
```

**To deploy manually** (e.g. to recover from a failed CI run):
```bash
pip install ".[docs,cypari2,flint]"
mkdocs gh-deploy --force
```

**Hosting:** the site is served by GitHub Pages from the `gh-pages` branch at
`https://antieau.github.io/snforacle/`. GitHub Pages must be enabled in the
repository settings (Settings → Pages → Source: `gh-pages` branch, `/ root`).

**Alternative hosting:** `mkdocs build` produces a self-contained `site/`
directory that can be served from any static file host (own server via rsync,
Netlify, Cloudflare Pages, AWS S3, etc.). Replace the `mkdocs gh-deploy` step
in CI with whatever upload command your host requires.

## Future: versioned docs with mike

When there are multiple supported release series (e.g. 1.x and 2.x), replace
`mkdocs gh-deploy` with [`mike`](https://github.com/jimporter/mike):

```bash
pip install mike
mike deploy --push --update-aliases 1.0 latest
```

`mike` maintains named versions side-by-side on `gh-pages` and the Material
theme provides a built-in version picker. Until then, a single unversioned
deployment per release is sufficient.

## Future work

### Sparse matrix support for MAGMA backend

MAGMA has a native sparse matrix module (`SparseMatrix`) that could give
significant performance gains for large sparse inputs (e.g. boundary matrices
from simplicial complexes). Currently the CLI backend always builds a dense
`Matrix(Integers(), [...])`, discarding any sparsity information that the caller
provided.

To implement: detect when the input `SparseIntegerMatrix` has density below some
threshold, emit `SparseMatrix(Integers(), nrows, ncols, [...])` in the MAGMA
script instead of `Matrix`, and add a conversion step if MAGMA returns a dense
result. The same idea applies to the `SparseFFMatrix` (F_p) and
`SparsePolyMatrix` (F_p[x]) input formats with MAGMA's corresponding sparse
types.

## cypari2 and CI

`cypari2` requires PARI/GP at compile time. Pre-built wheels are available on
PyPI for Python 3.12 on Linux x86_64; older versions must build from source and
therefore need `gp` on the runner. If multi-version testing is ever needed,
add this step before `pip install`:

```yaml
- name: Install PARI/GP
  run: sudo apt-get install -y libpari-dev pari-gp
```
