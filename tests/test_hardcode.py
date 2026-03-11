"""Hardcoded regression tests for all three matrix domains.

Each case was generated with a fixed random seed (1337) and the expected
outputs were computed by the pure_python backend (always available, verified
correct for small matrices against MAGMA and other backends).

30 cases total: 10 ZZ, 10 Fp, 10 Fp[x].
The Fp and Fp[x] cases use a variety of small primes: 2, 3, 5, 7, 11, 13.
"""

from __future__ import annotations

import pytest

from snforacle import (
    # ZZ
    elementary_divisors,
    hermite_normal_form,
    smith_normal_form,
    # Fp
    ff_hermite_normal_form,
    ff_rank,
    ff_smith_normal_form,
    # Fp[x]
    poly_elementary_divisors,
    poly_hermite_normal_form,
    poly_smith_normal_form,
)

# ---------------------------------------------------------------------------
# ZZ hardcoded cases
# Matrices generated with random.Random(1337), entries in [-5, 5].
# ---------------------------------------------------------------------------

_ZZ_CASES = [
    # (nrows, ncols, entries, invariant_factors, snf_entries, hnf_entries)
    (2, 2,
     [[4, 3], [0, 4]],
     [1, 16],
     [[1, 0], [0, 16]],
     [[4, 3], [0, 4]]),
    (3, 3,
     [[4, -3, 0], [1, 5, 0], [-1, 1, -2]],
     [1, 1, 46],
     [[1, 0, 0], [0, 1, 0], [0, 0, 46]],
     [[1, 0, 40], [0, 1, 38], [0, 0, 46]]),
    (2, 3,
     [[5, 0, -4], [1, 5, 1]],
     [1, 1],
     [[1, 0, 0], [0, 1, 0]],
     [[1, 5, 1], [0, 25, 9]]),
    (3, 2,
     [[-4, 3], [5, 0], [1, 1]],
     [1, 1],
     [[1, 0], [0, 1], [0, 0]],
     [[1, 0], [0, 1], [0, 0]]),
    (4, 4,
     [[-1, 5, -5, 0], [4, -4, 2, -2], [-3, 0, 5, 4], [2, 4, -1, -2]],
     [1, 1, 1, 174],
     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 174]],
     [[1, 0, 0, 52], [0, 1, 0, 112], [0, 0, 1, 32], [0, 0, 0, 174]]),
    (2, 4,
     [[3, -4, 5, 3], [4, 4, -5, -4]],
     [1, 1],
     [[1, 0, 0, 0], [0, 1, 0, 0]],
     [[1, 8, -10, -7], [0, 28, -35, -24]]),
    (4, 2,
     [[1, 5], [4, -5], [5, -5], [5, -2]],
     [1, 1],
     [[1, 0], [0, 1], [0, 0], [0, 0]],
     [[1, 0], [0, 1], [0, 0], [0, 0]]),
    (3, 4,
     [[-2, -4, -1, -2], [2, 4, 0, -5], [4, 0, 1, 4]],
     [1, 1, 2],
     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0]],
     [[2, 4, 0, -5], [0, 8, 0, -7], [0, 0, 1, 7]]),
    (4, 3,
     [[-2, 3, -5], [3, 5, -1], [-2, 4, -3], [3, 3, 3]],
     [1, 1, 1],
     [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
     [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),
    (3, 3,
     [[-3, 5, -5], [2, 4, 2], [-5, 3, 2]],
     [1, 1, 206],
     [[1, 0, 0], [0, 1, 0], [0, 0, 206]],
     [[1, 1, 48], [0, 2, 9], [0, 0, 103]]),
]


@pytest.mark.parametrize("backend", ["pure_python", "cypari2", "flint", "sage", "magma"])
@pytest.mark.parametrize("nrows,ncols,entries,inv_factors,snf_expected,hnf_expected",
                         _ZZ_CASES,
                         ids=[f"zz{i}" for i in range(len(_ZZ_CASES))])
def test_zz_snf(backend, nrows, ncols, entries, inv_factors, snf_expected, hnf_expected):
    inp = {"format": "dense", "nrows": nrows, "ncols": ncols, "entries": entries}
    try:
        res = smith_normal_form(inp, backend=backend)
    except RuntimeError as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.invariant_factors == inv_factors
    assert res.smith_normal_form.entries == snf_expected


@pytest.mark.parametrize("backend", ["pure_python", "flint", "sage", "magma"])
@pytest.mark.parametrize("nrows,ncols,entries,inv_factors,snf_expected,hnf_expected",
                         _ZZ_CASES,
                         ids=[f"zz{i}" for i in range(len(_ZZ_CASES))])
def test_zz_hnf(backend, nrows, ncols, entries, inv_factors, snf_expected, hnf_expected):
    inp = {"format": "dense", "nrows": nrows, "ncols": ncols, "entries": entries}
    try:
        res = hermite_normal_form(inp, backend=backend)
    except (RuntimeError, NotImplementedError) as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.hermite_normal_form.entries == hnf_expected


@pytest.mark.parametrize("backend", ["pure_python", "cypari2", "flint", "sage", "magma"])
@pytest.mark.parametrize("nrows,ncols,entries,inv_factors,snf_expected,hnf_expected",
                         _ZZ_CASES,
                         ids=[f"zz{i}" for i in range(len(_ZZ_CASES))])
def test_zz_elementary_divisors(backend, nrows, ncols, entries, inv_factors, snf_expected, hnf_expected):
    inp = {"format": "dense", "nrows": nrows, "ncols": ncols, "entries": entries}
    try:
        res = elementary_divisors(inp, backend=backend)
    except RuntimeError as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.elementary_divisors == inv_factors


# ---------------------------------------------------------------------------
# Fp hardcoded cases
# Primes: 2, 3, 5, 7, 11, 13, 2, 3, 5, 7
# ---------------------------------------------------------------------------

_FF_CASES = [
    # (p, nrows, ncols, entries, rank, snf_entries, hnf_entries)
    (2, 2, 2,
     [[1, 1], [1, 0]],
     2,
     [[1, 0], [0, 1]],
     [[1, 0], [0, 1]]),
    (3, 3, 3,
     [[2, 0, 0], [0, 1, 0], [0, 1, 2]],
     3,
     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
     [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    (5, 2, 3,
     [[2, 0, 4], [1, 3, 2]],
     2,
     [[1, 0, 0], [0, 1, 0]],
     [[1, 0, 2], [0, 1, 0]]),
    (7, 3, 2,
     [[4, 5], [0, 3], [2, 5]],
     2,
     [[1, 0], [0, 1], [0, 0]],
     [[1, 0], [0, 1], [0, 0]]),
    (11, 4, 4,
     [[2, 0, 10, 7], [5, 6, 1, 3], [5, 10, 10, 10], [9, 4, 3, 3]],
     4,
     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    (13, 2, 4,
     [[2, 7, 6, 9], [6, 8, 4, 2]],
     2,
     [[1, 0, 0, 0], [0, 1, 0, 0]],
     [[1, 10, 0, 1], [0, 0, 1, 12]]),
    (2, 4, 2,
     [[0, 1], [1, 0], [1, 0], [1, 1]],
     2,
     [[1, 0], [0, 1], [0, 0], [0, 0]],
     [[1, 0], [0, 1], [0, 0], [0, 0]]),
    (3, 3, 4,
     [[0, 0, 2, 2], [0, 2, 1, 1], [2, 1, 1, 1]],
     3,
     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]]),
    (5, 4, 3,
     [[1, 2, 3], [4, 3, 3], [1, 4, 2], [1, 3, 0]],
     3,
     [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
     [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),
    (7, 3, 3,
     [[3, 5, 0], [2, 0, 6], [2, 3, 4]],
     3,
     [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
     [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
]


@pytest.mark.parametrize("backend", ["pure_python", "flint", "sage", "magma"])
@pytest.mark.parametrize("p,nrows,ncols,entries,rank,snf_expected,hnf_expected",
                         _FF_CASES,
                         ids=[f"ff{i}" for i in range(len(_FF_CASES))])
def test_ff_snf(backend, p, nrows, ncols, entries, rank, snf_expected, hnf_expected):
    inp = {"format": "dense_ff", "nrows": nrows, "ncols": ncols, "p": p, "entries": entries}
    try:
        res = ff_smith_normal_form(inp, backend=backend)
    except RuntimeError as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.rank == rank
    assert res.smith_normal_form.entries == snf_expected


@pytest.mark.parametrize("backend", ["pure_python", "flint", "sage", "magma"])
@pytest.mark.parametrize("p,nrows,ncols,entries,rank,snf_expected,hnf_expected",
                         _FF_CASES,
                         ids=[f"ff{i}" for i in range(len(_FF_CASES))])
def test_ff_hnf(backend, p, nrows, ncols, entries, rank, snf_expected, hnf_expected):
    inp = {"format": "dense_ff", "nrows": nrows, "ncols": ncols, "p": p, "entries": entries}
    try:
        res = ff_hermite_normal_form(inp, backend=backend)
    except RuntimeError as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.hermite_normal_form.entries == hnf_expected


@pytest.mark.parametrize("backend", ["pure_python", "flint", "sage", "magma"])
@pytest.mark.parametrize("p,nrows,ncols,entries,rank,snf_expected,hnf_expected",
                         _FF_CASES,
                         ids=[f"ff{i}" for i in range(len(_FF_CASES))])
def test_ff_rank(backend, p, nrows, ncols, entries, rank, snf_expected, hnf_expected):
    inp = {"format": "dense_ff", "nrows": nrows, "ncols": ncols, "p": p, "entries": entries}
    try:
        res = ff_rank(inp, backend=backend)
    except RuntimeError as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.rank == rank


# ---------------------------------------------------------------------------
# Fp[x] hardcoded cases
# Primes: 2, 3, 5, 7, 11, 2, 3, 5, 7, 11
# Polynomials: coefficient lists [c0, c1, ...], constant-term first, [] = zero.
# ---------------------------------------------------------------------------

_POLY_CASES = [
    # (p, nrows, ncols, entries, invariant_factors, snf_entries, hnf_entries)
    (2, 2, 2,
     [[[1], [1, 0, 1]], [[], [0, 1]]],
     [[1], [0, 1]],
     [[[1], []], [[], [0, 1]]],
     [[[1], [1]], [[], [0, 1]]]),
    (3, 2, 2,
     [[[0, 1, 2], [1, 2]], [[2], [0, 2, 2]]],
     [[1], [1, 2, 2, 0, 1]],
     [[[1], []], [[], [1, 2, 2, 0, 1]]],
     [[[1], [0, 1, 1]], [[], [1, 2, 2, 0, 1]]]),
    (5, 3, 3,
     [[[0, 4, 3], [0, 1], [3]], [[4], [3], [1]], [[1], [], [3, 3, 3]]],
     [[1], [1], [3, 0, 3, 3, 1]],
     [[[1], [], []], [[], [1], []], [[], [], [3, 0, 3, 3, 1]]],
     [[[1], [], [3, 3, 3]], [[], [1], [3, 1, 1]], [[], [], [3, 0, 3, 3, 1]]]),
    (7, 2, 3,
     [[[2, 6, 2], [5], [2, 6]], [[0, 6, 1], [5], [2]]],
     [[1], [1]],
     [[[1], [], []], [[], [1], []]],
     [[[1], [3, 2], [4, 1, 6]], [[], [2, 0, 1], [5, 0, 3, 3]]]),
    (11, 3, 2,
     [[[10, 3, 2], [3, 1, 1]], [[8], [6, 9, 9]], [[], [3, 10]]],
     [[1], [1]],
     [[[1], []], [[], [1]], [[], []]],
     [[[1], []], [[], [1]], [[], []]]),
    (2, 2, 2,
     [[[1, 1], []], [[], [0, 1, 1]]],
     [[1, 1], [0, 1, 1]],
     [[[1, 1], []], [[], [0, 1, 1]]],
     [[[1, 1], []], [[], [0, 1, 1]]]),
    (3, 3, 3,
     [[[1], [0, 0, 2], [0, 2, 1]], [[2, 2], [0, 2], [1]], [[1, 1], [2], [0, 2]]],
     [[1], [1], [2, 1, 1, 2, 1]],
     [[[1], [], []], [[], [1], []], [[], [], [2, 1, 1, 2, 1]]],
     [[[1], [], [1, 0, 1, 2]], [[], [1], [0, 0, 2, 2]], [[], [], [2, 1, 1, 2, 1]]]),
    (5, 2, 3,
     [[[], [2], [4, 3]], [[3, 1], [3], [1, 1]]],
     [[1], [1]],
     [[[1], [], []], [[], [1], []]],
     [[[3, 1], [], [0, 4]], [[], [1], [2, 4]]]),
    (7, 3, 2,
     [[[1], [5, 4, 6]], [[3], [4]], [[1], [1, 3]]],
     [[1], [1]],
     [[[1], []], [[], [1]], [[], []]],
     [[[1], []], [[], [1]], [[], []]]),
    (11, 2, 2,
     [[[8, 8], [1, 10]], [[4], [7, 5, 5]]],
     [[1], [9, 8, 2, 1]],
     [[[1], []], [[], [9, 8, 2, 1]]],
     [[[1], [10, 4, 4]], [[], [9, 8, 2, 1]]]),
]


@pytest.mark.parametrize("backend", ["pure_python", "sage", "magma"])
@pytest.mark.parametrize(
    "p,nrows,ncols,entries,inv_factors,snf_expected,hnf_expected",
    _POLY_CASES,
    ids=[f"poly{i}" for i in range(len(_POLY_CASES))],
)
def test_poly_snf(backend, p, nrows, ncols, entries, inv_factors, snf_expected, hnf_expected):
    inp = {"format": "dense_poly", "nrows": nrows, "ncols": ncols, "p": p, "entries": entries}
    try:
        res = poly_smith_normal_form(inp, backend=backend)
    except RuntimeError as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.invariant_factors == inv_factors
    assert res.smith_normal_form.entries == snf_expected


@pytest.mark.parametrize("backend", ["pure_python", "sage", "magma"])
@pytest.mark.parametrize(
    "p,nrows,ncols,entries,inv_factors,snf_expected,hnf_expected",
    _POLY_CASES,
    ids=[f"poly{i}" for i in range(len(_POLY_CASES))],
)
def test_poly_hnf(backend, p, nrows, ncols, entries, inv_factors, snf_expected, hnf_expected):
    inp = {"format": "dense_poly", "nrows": nrows, "ncols": ncols, "p": p, "entries": entries}
    try:
        res = poly_hermite_normal_form(inp, backend=backend)
    except RuntimeError as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.hermite_normal_form.entries == hnf_expected


@pytest.mark.parametrize("backend", ["pure_python", "sage", "magma"])
@pytest.mark.parametrize(
    "p,nrows,ncols,entries,inv_factors,snf_expected,hnf_expected",
    _POLY_CASES,
    ids=[f"poly{i}" for i in range(len(_POLY_CASES))],
)
def test_poly_elementary_divisors(backend, p, nrows, ncols, entries, inv_factors, snf_expected, hnf_expected):
    inp = {"format": "dense_poly", "nrows": nrows, "ncols": ncols, "p": p, "entries": entries}
    try:
        res = poly_elementary_divisors(inp, backend=backend)
    except RuntimeError as e:
        if "not available" in str(e) or "not installed" in str(e):
            pytest.skip(f"{backend} not available")
        raise
    assert res.elementary_divisors == inv_factors
