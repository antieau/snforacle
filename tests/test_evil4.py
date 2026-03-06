"""Evil edge cases for all operations and all backends.

Tests elementary_divisors, hermite_normal_form, hermite_normal_form_with_transform,
smith_normal_form, and smith_normal_form_with_transforms across every backend that
supports the operation and is currently available.
"""

from __future__ import annotations

import shutil
import pytest

from snforacle import (
    elementary_divisors,
    hermite_normal_form,
    hermite_normal_form_with_transform,
    smith_normal_form,
    smith_normal_form_with_transforms,
)


# ---------------------------------------------------------------------------
# Backend availability (detected once at module import time)
# ---------------------------------------------------------------------------

def _avail(name: str) -> bool:
    """Return True only if the backend is installed AND works (catches sandbox issues)."""
    if name == "pure_python":
        return True
    if name in ("sage", "magma"):
        if not shutil.which(name):
            return False
        # Trial run: a permission error or other failure means the CLI is broken here.
        try:
            from snforacle import smith_normal_form  # noqa: PLC0415
            smith_normal_form(
                {"format": "dense", "nrows": 1, "ncols": 1, "entries": [[1]]},
                backend=name,
            )
            return True
        except Exception:
            return False
    try:
        __import__(name)
        return True
    except ImportError:
        return False


_ALL = ["cypari2", "flint", "sage", "magma", "pure_python"]
_ED_BACKENDS    = [b for b in _ALL if _avail(b)]
_HNF_BACKENDS   = [b for b in _ALL if b != "cypari2" and _avail(b)]
_HNF_T_BACKENDS = [b for b in _ALL if b not in ("cypari2", "flint") and _avail(b)]
_SNF_BACKENDS   = [b for b in _ALL if _avail(b)]
_SNF_T_BACKENDS = [b for b in _ALL if b != "flint" and _avail(b)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense(entries: list[list[int]], nrows: int, ncols: int) -> dict:
    return {"format": "dense", "nrows": nrows, "ncols": ncols, "entries": entries}


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    m, n = len(A), len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _verify_hnf(H: list[list[int]], nrows: int, ncols: int) -> None:
    """Assert H has valid HNF structure: positive pivots and upper-trapezoidal."""
    prev = -1
    for i in range(nrows):
        pj = next((j for j in range(ncols) if H[i][j] != 0), None)
        if pj is None:
            for k in range(i, nrows):
                assert all(H[k][j] == 0 for j in range(ncols)), (
                    f"non-zero entry in row {k} after zero row {i}"
                )
            return
        assert pj > prev, f"pivot col {pj} ≤ previous {prev} at row {i}"
        assert H[i][pj] > 0, f"pivot H[{i}][{pj}] = {H[i][pj]} not positive"
        for j in range(pj):
            assert H[i][j] == 0, f"H[{i}][{j}] = {H[i][j]} ≠ 0 before pivot"
        for k in range(i + 1, nrows):
            assert H[k][pj] == 0, f"H[{k}][{pj}] ≠ 0 below pivot at row {i}"
        prev = pj


# ---------------------------------------------------------------------------
# Evil test cases: (tag, nrows, ncols, entries, expected_inv_factors or None)
#
# expected_inv_factors is the expected SNF invariant factors (= elementary
# divisors).  None means "no hardcoded expectation; rely on cross-backend
# consistency only".
# ---------------------------------------------------------------------------

_CASES = [
    # Trivial / degenerate
    ("zero_1x1",     1, 1, [[0]],                                             []),
    ("zero_3x3",     3, 3, [[0,0,0],[0,0,0],[0,0,0]],                        []),
    ("id_1x1",       1, 1, [[1]],                                             [1]),
    ("id_3x3",       3, 3, [[1,0,0],[0,1,0],[0,0,1]],                        [1,1,1]),
    # Negative entries — SNF always has non-negative invariant factors
    ("neg_1x1",      1, 1, [[-1]],                                            [1]),
    ("neg_diag_2x2", 2, 2, [[-1,0],[0,-2]],                                   [1,2]),
    # Standard textbook example
    ("std_3x3",      3, 3, [[2,4,4],[-6,6,12],[10,-4,-16]],                   [2,6,12]),
    # Diagonal matrices (already in SNF if divisibility holds; otherwise not)
    ("diag_2_4",     2, 2, [[2,0],[0,4]],                                     [2,4]),
    ("diag_6_4",     2, 2, [[6,0],[0,4]],                                     [2,12]),
    # Rank-deficient square matrices
    ("rank1_3x3",    3, 3, [[1,2,3],[2,4,6],[3,6,9]],                        [1]),
    ("rank2_3x3",    3, 3, [[2,4,4],[-6,6,12],[4,8,8]],                      [2,6]),
    # Non-square matrices
    ("2x3",          2, 3, [[1,2,3],[4,5,6]],                                [1,3]),
    ("3x2",          3, 2, [[1,4],[2,5],[3,6]],                              [1,3]),
    ("1x5_gcd3",     1, 5, [[3,6,9,12,15]],                                  [3]),
    ("5x1_gcd3",     5, 1, [[3],[6],[9],[12],[15]],                           [3]),
    ("3x4_arith",    3, 4, [[1,2,3,4],[5,6,7,8],[9,10,11,12]],               [1,4]),
    ("5x2_rank2",    5, 2, [[1,0],[0,1],[0,0],[0,0],[0,0]],                  [1,1]),
    ("2x5_zero",     2, 5, [[0,0,0,0,0],[0,0,0,0,0]],                        []),
    # Small det
    ("det2_2x2",     2, 2, [[1,2],[3,4]],                                     [1,2]),
    # Unimodular (SNF = identity, all invariant factors are 1)
    ("unimod_3x3",   3, 3, [[1,2,3],[0,1,4],[0,0,1]],                        [1,1,1]),
    # Large prime entry
    ("mersenne_1x1", 1, 1, [[2**61-1]],                                       [2**61-1]),
]

_IDS = [c[0] for c in _CASES]


# ---------------------------------------------------------------------------
# 1. Elementary divisors
# ---------------------------------------------------------------------------

class TestEvilAllBackendsED:
    """ED on every evil case with every available backend."""

    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_ed(self, tag, nrows, ncols, entries, expected_inv):
        assert _ED_BACKENDS, "No ED backends available"
        inp = _dense(entries, nrows, ncols)
        ref_b = _ED_BACKENDS[0]
        ref = elementary_divisors(inp, backend=ref_b)

        if expected_inv is not None:
            assert ref.elementary_divisors == expected_inv, (
                f"{tag}: expected {expected_inv}, {ref_b} returned {ref.elementary_divisors}"
            )

        for b in _ED_BACKENDS[1:]:
            result = elementary_divisors(inp, backend=b)
            assert result.elementary_divisors == ref.elementary_divisors, (
                f"{tag}: {ref_b}={ref.elementary_divisors} vs {b}={result.elementary_divisors}"
            )


# ---------------------------------------------------------------------------
# 2. Hermite Normal Form
# ---------------------------------------------------------------------------

class TestEvilAllBackendsHNF:
    """HNF on every evil case: structural properties + cross-backend consistency."""

    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_hnf(self, tag, nrows, ncols, entries, expected_inv):
        assert _HNF_BACKENDS, "No HNF backends available"
        inp = _dense(entries, nrows, ncols)
        ref_b = _HNF_BACKENDS[0]
        ref = hermite_normal_form(inp, backend=ref_b)
        H_ref = ref.hermite_normal_form.entries

        _verify_hnf(H_ref, nrows, ncols)

        for b in _HNF_BACKENDS[1:]:
            result = hermite_normal_form(inp, backend=b)
            H = result.hermite_normal_form.entries
            assert H == H_ref, (
                f"{tag}: {ref_b} vs {b} HNF differ:\n  {ref_b}: {H_ref}\n  {b}: {H}"
            )


# ---------------------------------------------------------------------------
# 3. HNF with transform
# ---------------------------------------------------------------------------

class TestEvilAllBackendsHNFWithTransform:
    """HNF + left unimodular transform: U@M == H and H agrees with HNF-only backends."""

    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_hnf_transform(self, tag, nrows, ncols, entries, expected_inv):
        if not _HNF_T_BACKENDS:
            pytest.skip("No HNF-transform backends available")
        inp = _dense(entries, nrows, ncols)

        # Ground-truth HNF from the HNF-only test
        ref_hnf = (
            hermite_normal_form(inp, backend=_HNF_BACKENDS[0]).hermite_normal_form.entries
            if _HNF_BACKENDS else None
        )

        for b in _HNF_T_BACKENDS:
            result = hermite_normal_form_with_transform(inp, backend=b)
            H = result.hermite_normal_form.entries
            U = result.left_transform.entries

            _verify_hnf(H, nrows, ncols)

            # Transform identity: U @ M == H
            assert _mat_mul(U, entries) == H, (
                f"{tag}: U@M ≠ H for backend {b}"
            )

            # HNF agrees with HNF-only backends
            if ref_hnf is not None:
                assert H == ref_hnf, (
                    f"{tag}: {b} HNF differs from {_HNF_BACKENDS[0]} HNF"
                )


# ---------------------------------------------------------------------------
# 4. Smith Normal Form
# ---------------------------------------------------------------------------

class TestEvilAllBackendsSNF:
    """SNF cross-backend consistency and expected-value checks."""

    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_snf(self, tag, nrows, ncols, entries, expected_inv):
        assert _SNF_BACKENDS, "No SNF backends available"
        inp = _dense(entries, nrows, ncols)
        ref_b = _SNF_BACKENDS[0]
        ref = smith_normal_form(inp, backend=ref_b)

        if expected_inv is not None:
            assert ref.invariant_factors == expected_inv, (
                f"{tag}: expected {expected_inv}, {ref_b} returned {ref.invariant_factors}"
            )

        for b in _SNF_BACKENDS[1:]:
            result = smith_normal_form(inp, backend=b)
            assert result.invariant_factors == ref.invariant_factors, (
                f"{tag}: {ref_b}={ref.invariant_factors} vs {b}={result.invariant_factors}"
            )
            assert result.smith_normal_form.entries == ref.smith_normal_form.entries, (
                f"{tag}: {ref_b} vs {b} SNF matrix entries differ"
            )


# ---------------------------------------------------------------------------
# 5. SNF with transforms
# ---------------------------------------------------------------------------

class TestEvilAllBackendsSNFWithTransforms:
    """SNF + transforms: invariant factors match and U@M@V == SNF for every backend."""

    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_snf_transforms(self, tag, nrows, ncols, entries, expected_inv):
        assert _SNF_T_BACKENDS, "No SNF-transform backends available"
        inp = _dense(entries, nrows, ncols)

        ref_inv = (
            smith_normal_form(inp, backend=_SNF_BACKENDS[0]).invariant_factors
            if _SNF_BACKENDS else None
        )

        for b in _SNF_T_BACKENDS:
            result = smith_normal_form_with_transforms(inp, backend=b)

            if ref_inv is not None:
                assert result.invariant_factors == ref_inv, (
                    f"{tag}: {b} invariant_factors={result.invariant_factors} ≠ ref={ref_inv}"
                )

            U = result.left_transform.entries
            V = result.right_transform.entries
            S = result.smith_normal_form.entries
            assert _mat_mul(_mat_mul(U, entries), V) == S, (
                f"{tag}: U@M@V ≠ SNF for backend {b}"
            )
