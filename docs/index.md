# snforacle

[![CI](https://github.com/antieau/snforacle/actions/workflows/ci.yml/badge.svg)](https://github.com/antieau/snforacle/actions/workflows/ci.yml)

**snforacle** computes the [Smith normal form](https://en.wikipedia.org/wiki/Smith_normal_form) (SNF), [Hermite normal form](https://en.wikipedia.org/wiki/Hermite_normal_form) (HNF), and elementary divisors of integer matrices, polynomial matrices over F_p[x], and matrices over finite fields F_p — all through a single, uniform Python API, regardless of which backend does the actual computation.

## What is the Smith Normal Form?

The **Smith normal form** of an integer matrix M is the unique diagonal matrix D = diag(d₁, d₂, …, dᵣ) with d₁ | d₂ | … | dᵣ such that D = U · M · V for some invertible integer matrices U, V. The diagonal entries are the **invariant factors** of M; they appear in homology computations, lattice problems, and number theory.

The **Hermite Normal Form** (row HNF) is the unique upper-triangular matrix H with positive pivots (smallest to largest) satisfying H = U · M for some unimodular integer matrix U.

The **elementary divisors** are the non-zero diagonal entries of the SNF, returned in non-decreasing order.

## Demo

<div id="snf-demo"></div>

## AI Disclaimer

This package was authored almost entirely by [Claude Code](https://claude.ai/). It is part of
the codebase being developed by [Ben Antieau](https://antieau.github.io/) and [Achim Krause](https://www.akrause.xyz/) for their various projects. The human in
the picture has caused CC to create extensive testing which compares the outputs across different
computer algebra systems. He is confident that this software is at least as bug-free as is standard
for mathematical software, and he thinks it is likely much more bug-free.

Testing included 1000s of random matrices across the available backends. In one case, we discovered
a bug in FLINT, which has now been fixed; see [flint#2592](https://github.com/flintlib/flint/issues/2592).
Ben will write more about the process of creating this package elsewhere.


## Backends

### Integer matrices

| Backend | Requires | SNF | SNF+T | HNF | HNF+T | Elem. Div. |
|---------|----------|-----|-------|-----|-------|-----------:|
| `cypari2` *(default)* | `pip install snforacle[cypari2]` | yes | yes | no | no | yes |
| `flint` | `pip install snforacle[flint]` | yes | no | yes | no | yes |
| `sage` | SageMath on PATH | yes | yes | yes | yes | yes |
| `magma` | MAGMA on PATH | yes | yes | yes | yes | yes |
| `pure_python` | none (stdlib only) | yes | yes | yes | yes | yes |

### Finite-field matrices (F_p)

| Backend | Requires | SNF | SNF+T | HNF | HNF+T | Rank |
|---------|----------|-----|-------|-----|-------|-----:|
| `flint` *(default)* | `pip install snforacle[flint]` | yes | yes* | yes | yes* | yes |
| `pure_python` | none | yes | yes | yes | yes | yes |
| `sage` | SageMath on PATH | yes | yes | yes | yes | yes |
| `magma` | MAGMA on PATH | yes | yes | yes | yes | yes |

### Polynomial matrices (F_p[x])

| Backend | Requires | SNF | SNF+T | HNF | HNF+T | Elem. Div. |
|---------|----------|-----|-------|-----|-------|-----------:|
| `sage` *(default)* | SageMath on PATH | yes | yes | yes | yes | yes |
| `magma` | MAGMA on PATH | yes | yes | yes | yes | yes |
| `pure_python` | none | yes | yes | yes | yes | yes |

## Installation

```bash
# PARI/GP backend (recommended default for integer matrices)
pip install "snforacle[cypari2]"

# FLINT backend
pip install "snforacle[flint]"

# Both
pip install "snforacle[cypari2,flint]"

# SageMath and MAGMA backends need no pip package —
# install SageMath or MAGMA separately and ensure
# 'sage' / 'magma' is on your PATH.
```

See the [Quick Start](quickstart.md) guide to get running immediately, or the [API Reference](api/integer.md) for full function signatures.
