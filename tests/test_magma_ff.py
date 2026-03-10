"""Tests for the MAGMA finite-field backend and cross-backend consistency."""

from __future__ import annotations

import shutil

import pytest

from snforacle import (
    ff_hermite_normal_form,
    ff_hermite_normal_form_with_transform,
    ff_rank,
    ff_smith_normal_form,
    ff_smith_normal_form_with_transforms,
)
from snforacle.ff_schema import (
    FFHNFResult,
    FFHNFWithTransformResult,
    FFRankResult,
    FFSNFResult,
    FFSNFWithTransformsResult,
)
from snforacle.backends.pure_python_ff import mat_mul_ff

pytestmark = pytest.mark.skipif(
    shutil.which("magma") is None,
    reason="magma binary not found on PATH",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense(nrows: int, ncols: int, p: int, entries: list[list[int]]) -> dict:
    return {"format": "dense_ff", "nrows": nrows, "ncols": ncols, "p": p, "entries": entries}


def _identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _is_snf(snf: list[list[int]], nrows: int, ncols: int, rank: int) -> bool:
    for i in range(nrows):
        for j in range(ncols):
            expected = 1 if (i == j and i < rank) else 0
            if snf[i][j] != expected:
                return False
    return True


def _is_rref(H: list[list[int]], nrows: int, ncols: int) -> bool:
    pivot_col = -1
    for i in range(nrows):
        pc = next((j for j in range(ncols) if H[i][j] != 0), None)
        if pc is None:
            return all(H[ii][j] == 0 for ii in range(i + 1, nrows) for j in range(ncols))
        if pc <= pivot_col or H[i][pc] != 1:
            return False
        if any(H[ii][pc] != 0 for ii in range(nrows) if ii != i):
            return False
        pivot_col = pc
    return True


# ---------------------------------------------------------------------------
# SNF
# ---------------------------------------------------------------------------

class TestMagmaFFSNF:
    def test_identity_p5(self):
        result = ff_smith_normal_form(_dense(3, 3, 5, _identity(3)), backend="magma")
        assert result.rank == 3
        assert _is_snf(result.smith_normal_form.entries, 3, 3, 3)

    def test_zero_p7(self):
        result = ff_smith_normal_form(_dense(2, 2, 7, [[0, 0], [0, 0]]), backend="magma")
        assert result.rank == 0
        assert _is_snf(result.smith_normal_form.entries, 2, 2, 0)

    def test_rank1_p5(self):
        result = ff_smith_normal_form(_dense(2, 2, 5, [[1, 2], [2, 4]]), backend="magma")
        assert result.rank == 1
        assert _is_snf(result.smith_normal_form.entries, 2, 2, 1)

    def test_full_rank_3x3_p11(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        result = ff_smith_normal_form(_dense(3, 3, 11, M), backend="magma")
        assert result.rank == 3
        assert _is_snf(result.smith_normal_form.entries, 3, 3, 3)

    def test_1x1_unit_p7(self):
        result = ff_smith_normal_form(_dense(1, 1, 7, [[3]]), backend="magma")
        assert result.rank == 1
        assert result.smith_normal_form.entries == [[1]]

    def test_nonsquare_2x3_p11(self):
        M = [[1, 0, 2], [0, 1, 3]]
        result = ff_smith_normal_form(_dense(2, 3, 11, M), backend="magma")
        assert result.rank == 2

    def test_return_type(self):
        result = ff_smith_normal_form(_dense(2, 2, 5, [[1, 2], [3, 4]]), backend="magma")
        assert isinstance(result, FFSNFResult)

    def test_p2(self):
        M = [[1, 1], [0, 1]]
        result = ff_smith_normal_form(_dense(2, 2, 2, M), backend="magma")
        assert result.rank == 2


# ---------------------------------------------------------------------------
# SNF with transforms
# ---------------------------------------------------------------------------

class TestMagmaFFSNFWithTransforms:
    def _verify(self, M: list[list[int]], nrows: int, ncols: int, p: int) -> FFSNFWithTransformsResult:
        result = ff_smith_normal_form_with_transforms(
            _dense(nrows, ncols, p, M), backend="magma"
        )
        snf = result.smith_normal_form.entries
        U = result.left_transform.entries
        V = result.right_transform.entries
        assert _is_snf(snf, nrows, ncols, result.rank)
        UMV = mat_mul_ff(mat_mul_ff(U, M, p), V, p)
        assert UMV == snf, f"U@M@V={UMV} != snf={snf}"
        return result

    def test_identity_p5(self):
        self._verify(_identity(3), 3, 3, 5)

    def test_zero_p7(self):
        self._verify([[0, 0], [0, 0]], 2, 2, 7)

    def test_rank1_p5(self):
        self._verify([[1, 2], [2, 4]], 2, 2, 5)

    def test_full_rank_3x3_p11(self):
        self._verify([[1, 2, 3], [4, 5, 6], [7, 8, 10]], 3, 3, 11)

    def test_nonsquare_2x3_p11(self):
        self._verify([[1, 2, 3], [4, 5, 6]], 2, 3, 11)

    def test_nonsquare_3x2_p5(self):
        self._verify([[1, 2], [3, 4], [0, 1]], 3, 2, 5)

    def test_return_type(self):
        result = ff_smith_normal_form_with_transforms(
            _dense(2, 2, 7, [[1, 2], [3, 4]]), backend="magma"
        )
        assert isinstance(result, FFSNFWithTransformsResult)
        assert result.left_transform.nrows == 2
        assert result.right_transform.ncols == 2


# ---------------------------------------------------------------------------
# HNF (RREF)
# ---------------------------------------------------------------------------

class TestMagmaFFHNF:
    def test_identity_p5(self):
        result = ff_hermite_normal_form(_dense(3, 3, 5, _identity(3)), backend="magma")
        assert result.hermite_normal_form.entries == _identity(3)

    def test_zero_p7(self):
        result = ff_hermite_normal_form(_dense(2, 2, 7, [[0, 0], [0, 0]]), backend="magma")
        assert result.hermite_normal_form.entries == [[0, 0], [0, 0]]

    def test_full_rank_3x3_p11(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        result = ff_hermite_normal_form(_dense(3, 3, 11, M), backend="magma")
        H = result.hermite_normal_form.entries
        assert _is_rref(H, 3, 3)
        assert H == _identity(3)

    def test_rank1_p5(self):
        result = ff_hermite_normal_form(_dense(2, 2, 5, [[1, 2], [2, 4]]), backend="magma")
        H = result.hermite_normal_form.entries
        assert _is_rref(H, 2, 2)

    def test_nonsquare_2x4_p7(self):
        M = [[1, 0, 2, 3], [0, 1, 4, 2]]
        result = ff_hermite_normal_form(_dense(2, 4, 7, M), backend="magma")
        H = result.hermite_normal_form.entries
        assert _is_rref(H, 2, 4)

    def test_return_type(self):
        result = ff_hermite_normal_form(_dense(2, 2, 5, [[1, 2], [3, 4]]), backend="magma")
        assert isinstance(result, FFHNFResult)


# ---------------------------------------------------------------------------
# HNF with transform
# ---------------------------------------------------------------------------

class TestMagmaFFHNFWithTransform:
    def _verify(self, M: list[list[int]], nrows: int, ncols: int, p: int) -> FFHNFWithTransformResult:
        result = ff_hermite_normal_form_with_transform(
            _dense(nrows, ncols, p, M), backend="magma"
        )
        H = result.hermite_normal_form.entries
        U = result.left_transform.entries
        assert _is_rref(H, nrows, ncols)
        UH = mat_mul_ff(U, M, p)
        assert UH == H, f"U@M={UH} != H={H}"
        return result

    def test_identity_p5(self):
        self._verify(_identity(3), 3, 3, 5)

    def test_zero_p7(self):
        self._verify([[0, 0], [0, 0]], 2, 2, 7)

    def test_rank1_p5(self):
        self._verify([[1, 2], [2, 4]], 2, 2, 5)

    def test_full_rank_3x3_p11(self):
        self._verify([[1, 2, 3], [4, 5, 6], [7, 8, 10]], 3, 3, 11)

    def test_nonsquare_2x3_p11(self):
        self._verify([[1, 2, 3], [4, 5, 6]], 2, 3, 11)

    def test_return_type(self):
        result = ff_hermite_normal_form_with_transform(
            _dense(2, 2, 7, [[1, 2], [3, 4]]), backend="magma"
        )
        assert isinstance(result, FFHNFWithTransformResult)
        assert result.left_transform.nrows == 2


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------

class TestMagmaFFRank:
    def test_full_rank(self):
        result = ff_rank(_dense(3, 3, 7, _identity(3)), backend="magma")
        assert result.rank == 3

    def test_zero(self):
        result = ff_rank(_dense(2, 2, 5, [[0, 0], [0, 0]]), backend="magma")
        assert result.rank == 0

    def test_rank1(self):
        result = ff_rank(_dense(2, 2, 5, [[1, 2], [2, 4]]), backend="magma")
        assert result.rank == 1

    def test_return_type(self):
        result = ff_rank(_dense(2, 2, 7, [[1, 0], [0, 1]]), backend="magma")
        assert isinstance(result, FFRankResult)


# ---------------------------------------------------------------------------
# Cross-backend consistency: magma vs pure_python
# ---------------------------------------------------------------------------

class TestMagmaFFVsPurePython:
    """Verify magma and pure_python backends agree on all operations."""

    def _inp(self, nrows, ncols, p, entries):
        return _dense(nrows, ncols, p, entries)

    def _check_snf(self, nrows, ncols, p, entries):
        inp = self._inp(nrows, ncols, p, entries)
        r_magma = ff_smith_normal_form(inp, backend="magma")
        r_pp = ff_smith_normal_form(inp, backend="pure_python")
        assert r_magma.rank == r_pp.rank
        assert r_magma.smith_normal_form.entries == r_pp.smith_normal_form.entries

    def _check_hnf(self, nrows, ncols, p, entries):
        inp = self._inp(nrows, ncols, p, entries)
        r_magma = ff_hermite_normal_form(inp, backend="magma")
        r_pp = ff_hermite_normal_form(inp, backend="pure_python")
        assert r_magma.hermite_normal_form.entries == r_pp.hermite_normal_form.entries

    def test_identity_p5(self):
        self._check_snf(3, 3, 5, _identity(3))
        self._check_hnf(3, 3, 5, _identity(3))

    def test_zero_p7(self):
        self._check_snf(2, 2, 7, [[0, 0], [0, 0]])
        self._check_hnf(2, 2, 7, [[0, 0], [0, 0]])

    def test_rank1_p5(self):
        self._check_snf(2, 2, 5, [[1, 2], [2, 4]])
        self._check_hnf(2, 2, 5, [[1, 2], [2, 4]])

    def test_full_rank_3x3_p11(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        self._check_snf(3, 3, 11, M)
        self._check_hnf(3, 3, 11, M)

    def test_nonsquare_2x3_p11(self):
        M = [[1, 2, 3], [4, 5, 6]]
        self._check_snf(2, 3, 11, M)
        self._check_hnf(2, 3, 11, M)

    def test_nonsquare_3x2_p5(self):
        M = [[1, 2], [3, 4], [0, 1]]
        self._check_snf(3, 2, 5, M)
        self._check_hnf(3, 2, 5, M)

    def test_p2(self):
        M = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        self._check_snf(3, 3, 2, M)
        self._check_hnf(3, 3, 2, M)

    def test_large_p(self):
        M = [[1, 2], [3, 4]]
        self._check_snf(2, 2, 101, M)

    def test_rank_deficient(self):
        M = [[1, 2, 3], [2, 4, 6], [3, 6, 2]]  # 9 → 9%7=2
        self._check_snf(3, 3, 7, M)
        self._check_hnf(3, 3, 7, M)
