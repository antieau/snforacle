"""Cross-backend consistency tests with MAGMA as the reference implementation.

All three matrix domains are covered:

  ZZ   — integer matrices (5 operations: SNF, SNF+T, HNF, HNF+T, ED)
  Fp   — finite-field matrices (5 operations: SNF, SNF+T, HNF, HNF+T, rank)
  Fp[x]— polynomial matrices (5 operations: SNF, SNF+T, HNF, HNF+T, ED)

MAGMA is treated as the ground-truth reference because it is widely regarded
as the most reliable computer algebra system for these computations.  The
entire module is skipped when MAGMA is not on PATH.

Matrix sizes and counts
-----------------------
ZZ   : 30 matrices, shapes in [10, 30] × [10, 30], entries in [-10, 10].
       pure_python is additionally compared on every 10th matrix with
       max(nrows, ncols) ≤ PP_MAX=12 to guard against integer blow-up.
Fp   : 30 matrices, shapes in [5, 20] × [5, 20], primes from a fixed cycle.
       pure_python compared on every 5th matrix.
Fp[x]: 15 matrices, shapes in [2, 5] × [2, 5], degree ≤ 2, primes cycled.
       pure_python compared on every matrix (small enough to be fast).
"""

from __future__ import annotations

import random
import shutil

import pytest

from _mathelpers import assert_unimodular, assert_invertible_ff, assert_invertible_poly

from snforacle import (
    # ZZ
    elementary_divisors,
    hermite_normal_form,
    hermite_normal_form_with_transform,
    smith_normal_form,
    smith_normal_form_with_transforms,
    # Fp
    ff_hermite_normal_form,
    ff_hermite_normal_form_with_transform,
    ff_smith_normal_form,
    ff_smith_normal_form_with_transforms,
    ff_rank,
    # Fp[x]
    poly_elementary_divisors,
    poly_hermite_normal_form,
    poly_hermite_normal_form_with_transform,
    poly_smith_normal_form,
    poly_smith_normal_form_with_transforms,
)

pytestmark = pytest.mark.skipif(
    shutil.which("magma") is None,
    reason="MAGMA not on PATH",
)

# ---------------------------------------------------------------------------
# Availability flags for secondary backends
# ---------------------------------------------------------------------------

def _avail_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


_HAVE_CYPARI2  = _avail_import("cypari2")
_HAVE_FLINT    = _avail_import("flint")
_HAVE_SAGE     = shutil.which("sage") is not None

# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _dense_int(entries: list[list[int]]) -> dict:
    return {
        "format": "dense",
        "nrows": len(entries),
        "ncols": len(entries[0]) if entries else 0,
        "entries": entries,
    }


def _dense_ff(entries: list[list[int]], p: int) -> dict:
    return {
        "format": "dense_ff",
        "nrows": len(entries),
        "ncols": len(entries[0]) if entries else 0,
        "p": p,
        "entries": entries,
    }


def _dense_poly(entries: list[list[list[int]]], p: int) -> dict:
    return {
        "format": "dense_poly",
        "nrows": len(entries),
        "ncols": len(entries[0]) if entries else 0,
        "p": p,
        "entries": entries,
    }


def _mat_mul_int(
    A: list[list[int]], B: list[list[int]]
) -> list[list[int]]:
    m, k = len(A), len(A[0]) if A else 0
    n = len(B[0]) if B else 0
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)]
        for i in range(m)
    ]


def _mat_mul_ff(
    A: list[list[int]], B: list[list[int]], p: int
) -> list[list[int]]:
    m, k = len(A), len(A[0]) if A else 0
    n = len(B[0]) if B else 0
    return [
        [sum(A[i][t] * B[t][j] for t in range(k)) % p for j in range(n)]
        for i in range(m)
    ]


# ---------------------------------------------------------------------------
# Random matrix generators
# ---------------------------------------------------------------------------

_FF_PRIMES  = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
_POLY_PRIMES = [2, 3, 5, 7, 11]


def _gen_zz(n: int, seed: int = 42):
    rng = random.Random(seed)
    for i in range(n):
        nrows = rng.randint(10, 30)
        ncols = rng.randint(10, 30)
        entries = [
            [rng.randint(-10, 10) for _ in range(ncols)]
            for _ in range(nrows)
        ]
        yield i, nrows, ncols, entries


def _gen_ff(n: int, seed: int = 42):
    rng = random.Random(seed)
    for i in range(n):
        p = _FF_PRIMES[i % len(_FF_PRIMES)]
        nrows = rng.randint(5, 20)
        ncols = rng.randint(5, 20)
        entries = [
            [rng.randint(0, p - 1) for _ in range(ncols)]
            for _ in range(nrows)
        ]
        yield i, p, nrows, ncols, entries


def _rand_poly(p: int, max_deg: int, rng: random.Random) -> list[int]:
    deg = rng.randint(0, max_deg)
    coeffs = [rng.randint(0, p - 1) for _ in range(deg + 1)]
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs


def _gen_poly(n: int, seed: int = 42):
    rng = random.Random(seed)
    for i in range(n):
        p = _POLY_PRIMES[i % len(_POLY_PRIMES)]
        nrows = rng.randint(2, 5)
        ncols = rng.randint(2, 5)
        entries = [
            [_rand_poly(p, 2, rng) for _ in range(ncols)]
            for _ in range(nrows)
        ]
        yield i, p, nrows, ncols, entries


# ---------------------------------------------------------------------------
# ZZ cross-backend tests
# ---------------------------------------------------------------------------

PP_STRIDE = 10
PP_MAX    = 12


class TestZZCrossBackend:
    """MAGMA vs all available backends on 30 random integer matrices."""

    N = 30

    def test_snf(self):
        for i, nrows, ncols, entries in _gen_zz(self.N):
            inp     = _dense_int(entries)
            ref_inv = smith_normal_form(inp, backend="magma").invariant_factors

            if _HAVE_CYPARI2:
                got = smith_normal_form(inp, backend="cypari2").invariant_factors
                assert got == ref_inv, f"ZZ SNF matrix {i}: cypari2 ≠ magma"
            if _HAVE_FLINT:
                got = smith_normal_form(inp, backend="flint").invariant_factors
                assert got == ref_inv, f"ZZ SNF matrix {i}: flint ≠ magma"
            if _HAVE_SAGE:
                got = smith_normal_form(inp, backend="sage").invariant_factors
                assert got == ref_inv, f"ZZ SNF matrix {i}: sage ≠ magma"
            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                got = smith_normal_form(inp, backend="pure_python").invariant_factors
                assert got == ref_inv, f"ZZ SNF matrix {i}: pure_python ≠ magma"

    def test_snf_with_transforms(self):
        for i, nrows, ncols, entries in _gen_zz(self.N):
            inp     = _dense_int(entries)
            ref_inv = smith_normal_form(inp, backend="magma").invariant_factors

            for b in (["cypari2"] if _HAVE_CYPARI2 else []) + (["sage"] if _HAVE_SAGE else []):
                res = smith_normal_form_with_transforms(inp, backend=b)
                assert res.invariant_factors == ref_inv, (
                    f"ZZ SNF+T matrix {i}: {b} invariant_factors ≠ magma"
                )
                U = res.left_transform.entries
                V = res.right_transform.entries
                S = res.smith_normal_form.entries
                assert _mat_mul_int(_mat_mul_int(U, entries), V) == S, (
                    f"ZZ SNF+T matrix {i}: U@M@V ≠ SNF for {b}"
                )
                assert_unimodular(U, f"ZZ SNF+T matrix {i} {b} U")
                assert_unimodular(V, f"ZZ SNF+T matrix {i} {b} V")
            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                res = smith_normal_form_with_transforms(inp, backend="pure_python")
                assert res.invariant_factors == ref_inv, (
                    f"ZZ SNF+T matrix {i}: pure_python invariant_factors ≠ magma"
                )
                U = res.left_transform.entries
                V = res.right_transform.entries
                S = res.smith_normal_form.entries
                assert _mat_mul_int(_mat_mul_int(U, entries), V) == S, (
                    f"ZZ SNF+T matrix {i}: U@M@V ≠ SNF for pure_python"
                )
                assert_unimodular(U, f"ZZ SNF+T matrix {i} pure_python U")
                assert_unimodular(V, f"ZZ SNF+T matrix {i} pure_python V")

    def test_hnf(self):
        for i, nrows, ncols, entries in _gen_zz(self.N):
            inp     = _dense_int(entries)
            ref_hnf = hermite_normal_form(inp, backend="magma").hermite_normal_form.entries

            if _HAVE_FLINT:
                got = hermite_normal_form(inp, backend="flint").hermite_normal_form.entries
                assert got == ref_hnf, f"ZZ HNF matrix {i}: flint ≠ magma"
            if _HAVE_SAGE:
                got = hermite_normal_form(inp, backend="sage").hermite_normal_form.entries
                assert got == ref_hnf, f"ZZ HNF matrix {i}: sage ≠ magma"
            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                got = hermite_normal_form(inp, backend="pure_python").hermite_normal_form.entries
                assert got == ref_hnf, f"ZZ HNF matrix {i}: pure_python ≠ magma"

    def test_hnf_with_transform(self):
        for i, nrows, ncols, entries in _gen_zz(self.N):
            inp     = _dense_int(entries)
            ref_hnf = hermite_normal_form(inp, backend="magma").hermite_normal_form.entries

            for b in (["sage"] if _HAVE_SAGE else []):
                res = hermite_normal_form_with_transform(inp, backend=b)
                assert res.hermite_normal_form.entries == ref_hnf, (
                    f"ZZ HNF+T matrix {i}: {b} HNF ≠ magma"
                )
                assert _mat_mul_int(res.left_transform.entries, entries) == ref_hnf, (
                    f"ZZ HNF+T matrix {i}: U@M ≠ H for {b}"
                )
                assert_unimodular(res.left_transform.entries, f"ZZ HNF+T matrix {i} {b} U")
            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                res = hermite_normal_form_with_transform(inp, backend="pure_python")
                assert res.hermite_normal_form.entries == ref_hnf, (
                    f"ZZ HNF+T matrix {i}: pure_python HNF ≠ magma"
                )
                assert _mat_mul_int(res.left_transform.entries, entries) == ref_hnf, (
                    f"ZZ HNF+T matrix {i}: U@M ≠ H for pure_python"
                )
                assert_unimodular(res.left_transform.entries, f"ZZ HNF+T matrix {i} pure_python U")

    def test_elementary_divisors(self):
        for i, nrows, ncols, entries in _gen_zz(self.N):
            inp    = _dense_int(entries)
            ref_ed = elementary_divisors(inp, backend="magma").elementary_divisors

            if _HAVE_CYPARI2:
                got = elementary_divisors(inp, backend="cypari2").elementary_divisors
                assert got == ref_ed, f"ZZ ED matrix {i}: cypari2 ≠ magma"
            if _HAVE_FLINT:
                got = elementary_divisors(inp, backend="flint").elementary_divisors
                assert got == ref_ed, f"ZZ ED matrix {i}: flint ≠ magma"
            if _HAVE_SAGE:
                got = elementary_divisors(inp, backend="sage").elementary_divisors
                assert got == ref_ed, f"ZZ ED matrix {i}: sage ≠ magma"
            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                got = elementary_divisors(inp, backend="pure_python").elementary_divisors
                assert got == ref_ed, f"ZZ ED matrix {i}: pure_python ≠ magma"


# ---------------------------------------------------------------------------
# Fp cross-backend tests
# ---------------------------------------------------------------------------

FF_PP_STRIDE = 5


class TestFpCrossBackend:
    """MAGMA vs all available backends on 30 random Fp matrices."""

    N = 30

    def test_snf(self):
        for i, p, nrows, ncols, entries in _gen_ff(self.N):
            inp     = _dense_ff(entries, p)
            ref_snf = ff_smith_normal_form(inp, backend="magma")

            if _HAVE_FLINT:
                got = ff_smith_normal_form(inp, backend="flint")
                assert got.rank == ref_snf.rank, f"Fp SNF matrix {i} p={p}: flint rank ≠ magma"
                assert got.smith_normal_form.entries == ref_snf.smith_normal_form.entries, (
                    f"Fp SNF matrix {i} p={p}: flint SNF ≠ magma"
                )
            if _HAVE_SAGE:
                got = ff_smith_normal_form(inp, backend="sage")
                assert got.rank == ref_snf.rank, f"Fp SNF matrix {i} p={p}: sage rank ≠ magma"
            if i % FF_PP_STRIDE == 0:
                got = ff_smith_normal_form(inp, backend="pure_python")
                assert got.rank == ref_snf.rank, f"Fp SNF matrix {i} p={p}: pure_python rank ≠ magma"
                assert got.smith_normal_form.entries == ref_snf.smith_normal_form.entries, (
                    f"Fp SNF matrix {i} p={p}: pure_python SNF ≠ magma"
                )

    def test_snf_with_transforms(self):
        for i, p, nrows, ncols, entries in _gen_ff(self.N):
            inp      = _dense_ff(entries, p)
            ref_rank = ff_rank(inp, backend="magma").rank

            for b in (["sage"] if _HAVE_SAGE else []) + (
                ["pure_python"] if i % FF_PP_STRIDE == 0 else []
            ):
                res = ff_smith_normal_form_with_transforms(inp, backend=b)
                assert res.rank == ref_rank, (
                    f"Fp SNF+T matrix {i} p={p}: {b} rank ≠ magma"
                )
                U = res.left_transform.entries
                V = res.right_transform.entries
                S = res.smith_normal_form.entries
                assert _mat_mul_ff(_mat_mul_ff(U, entries, p), V, p) == S, (
                    f"Fp SNF+T matrix {i} p={p}: U@M@V ≠ SNF for {b}"
                )
                assert_invertible_ff(U, p, f"Fp SNF+T matrix {i} p={p} {b} U")
                assert_invertible_ff(V, p, f"Fp SNF+T matrix {i} p={p} {b} V")

    def test_hnf(self):
        for i, p, nrows, ncols, entries in _gen_ff(self.N):
            inp     = _dense_ff(entries, p)
            ref_hnf = ff_hermite_normal_form(inp, backend="magma").hermite_normal_form.entries

            if _HAVE_FLINT:
                got = ff_hermite_normal_form(inp, backend="flint").hermite_normal_form.entries
                assert got == ref_hnf, f"Fp HNF matrix {i} p={p}: flint ≠ magma"
            if _HAVE_SAGE:
                got = ff_hermite_normal_form(inp, backend="sage").hermite_normal_form.entries
                assert got == ref_hnf, f"Fp HNF matrix {i} p={p}: sage ≠ magma"
            if i % FF_PP_STRIDE == 0:
                got = ff_hermite_normal_form(inp, backend="pure_python").hermite_normal_form.entries
                assert got == ref_hnf, f"Fp HNF matrix {i} p={p}: pure_python ≠ magma"

    def test_hnf_with_transform(self):
        for i, p, nrows, ncols, entries in _gen_ff(self.N):
            inp     = _dense_ff(entries, p)
            ref_hnf = ff_hermite_normal_form(inp, backend="magma").hermite_normal_form.entries

            for b in (["sage"] if _HAVE_SAGE else []) + (
                ["pure_python"] if i % FF_PP_STRIDE == 0 else []
            ):
                res = ff_hermite_normal_form_with_transform(inp, backend=b)
                assert res.hermite_normal_form.entries == ref_hnf, (
                    f"Fp HNF+T matrix {i} p={p}: {b} HNF ≠ magma"
                )
                assert _mat_mul_ff(res.left_transform.entries, entries, p) == ref_hnf, (
                    f"Fp HNF+T matrix {i} p={p}: U@M ≠ H for {b}"
                )
                assert_invertible_ff(res.left_transform.entries, p, f"Fp HNF+T matrix {i} p={p} {b} U")

    def test_rank(self):
        for i, p, nrows, ncols, entries in _gen_ff(self.N):
            inp      = _dense_ff(entries, p)
            ref_rank = ff_rank(inp, backend="magma").rank

            if _HAVE_FLINT:
                got = ff_rank(inp, backend="flint").rank
                assert got == ref_rank, f"Fp rank matrix {i} p={p}: flint ≠ magma"
            if _HAVE_SAGE:
                got = ff_rank(inp, backend="sage").rank
                assert got == ref_rank, f"Fp rank matrix {i} p={p}: sage ≠ magma"
            if i % FF_PP_STRIDE == 0:
                got = ff_rank(inp, backend="pure_python").rank
                assert got == ref_rank, f"Fp rank matrix {i} p={p}: pure_python ≠ magma"


# ---------------------------------------------------------------------------
# Fp[x] cross-backend tests
# ---------------------------------------------------------------------------

class TestFpxCrossBackend:
    """MAGMA vs all available backends on 15 random Fp[x] matrices."""

    N = 15

    def test_snf(self):
        for i, p, nrows, ncols, entries in _gen_poly(self.N):
            inp     = _dense_poly(entries, p)
            ref_inv = poly_smith_normal_form(inp, backend="magma").invariant_factors

            if _HAVE_SAGE:
                got = poly_smith_normal_form(inp, backend="sage").invariant_factors
                assert got == ref_inv, f"Fp[x] SNF matrix {i} p={p}: sage ≠ magma"
            got = poly_smith_normal_form(inp, backend="pure_python").invariant_factors
            assert got == ref_inv, f"Fp[x] SNF matrix {i} p={p}: pure_python ≠ magma"

    def test_snf_with_transforms(self):
        for i, p, nrows, ncols, entries in _gen_poly(self.N):
            inp     = _dense_poly(entries, p)
            ref_inv = poly_smith_normal_form(inp, backend="magma").invariant_factors

            for b in (["sage"] if _HAVE_SAGE else []) + ["pure_python"]:
                res = poly_smith_normal_form_with_transforms(inp, backend=b)
                assert res.invariant_factors == ref_inv, (
                    f"Fp[x] SNF+T matrix {i} p={p}: {b} invariant_factors ≠ magma"
                )
                assert_invertible_poly(res.left_transform.entries, p, f"Fp[x] SNF+T matrix {i} p={p} {b} U")
                assert_invertible_poly(res.right_transform.entries, p, f"Fp[x] SNF+T matrix {i} p={p} {b} V")

    def test_hnf(self):
        for i, p, nrows, ncols, entries in _gen_poly(self.N):
            inp     = _dense_poly(entries, p)
            ref_hnf = poly_hermite_normal_form(inp, backend="magma").hermite_normal_form.entries

            if _HAVE_SAGE:
                got = poly_hermite_normal_form(inp, backend="sage").hermite_normal_form.entries
                assert got == ref_hnf, f"Fp[x] HNF matrix {i} p={p}: sage ≠ magma"
            got = poly_hermite_normal_form(inp, backend="pure_python").hermite_normal_form.entries
            assert got == ref_hnf, f"Fp[x] HNF matrix {i} p={p}: pure_python ≠ magma"

    def test_hnf_with_transform(self):
        for i, p, nrows, ncols, entries in _gen_poly(self.N):
            inp     = _dense_poly(entries, p)
            ref_hnf = poly_hermite_normal_form(inp, backend="magma").hermite_normal_form.entries

            for b in (["sage"] if _HAVE_SAGE else []) + ["pure_python"]:
                res = poly_hermite_normal_form_with_transform(inp, backend=b)
                assert res.hermite_normal_form.entries == ref_hnf, (
                    f"Fp[x] HNF+T matrix {i} p={p}: {b} HNF ≠ magma"
                )
                assert_invertible_poly(res.left_transform.entries, p, f"Fp[x] HNF+T matrix {i} p={p} {b} U")

    def test_elementary_divisors(self):
        for i, p, nrows, ncols, entries in _gen_poly(self.N):
            inp    = _dense_poly(entries, p)
            ref_ed = poly_elementary_divisors(inp, backend="magma").elementary_divisors

            if _HAVE_SAGE:
                got = poly_elementary_divisors(inp, backend="sage").elementary_divisors
                assert got == ref_ed, f"Fp[x] ED matrix {i} p={p}: sage ≠ magma"
            got = poly_elementary_divisors(inp, backend="pure_python").elementary_divisors
            assert got == ref_ed, f"Fp[x] ED matrix {i} p={p}: pure_python ≠ magma"
