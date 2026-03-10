"""Tests for the FLINT finite-field backend.

All tests are skipped if python-flint is not installed.
"""

from __future__ import annotations

import pytest

flint = pytest.importorskip("flint", reason="python-flint not installed")

from snforacle.backends.flint_ff import FlintFFBackend
from snforacle.backends.pure_python_ff import mat_mul_ff

BACKEND = FlintFFBackend()


def _identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _zero(nrows: int, ncols: int) -> list[list[int]]:
    return [[0] * ncols for _ in range(nrows)]


def _is_snf(snf: list[list[int]], nrows: int, ncols: int, rank: int) -> bool:
    for i in range(nrows):
        for j in range(ncols):
            expected = 1 if (i == j and i < rank) else 0
            if snf[i][j] != expected:
                return False
    return True


def _is_rref(H: list[list[int]], nrows: int, ncols: int, p: int) -> bool:
    pivot_col = -1
    for i in range(nrows):
        pc = None
        for j in range(ncols):
            if H[i][j] != 0:
                pc = j
                break
        if pc is None:
            for ii in range(i + 1, nrows):
                if any(H[ii][j] != 0 for j in range(ncols)):
                    return False
            break
        if pc <= pivot_col:
            return False
        if H[i][pc] != 1:
            return False
        for ii in range(nrows):
            if ii != i and H[ii][pc] != 0:
                return False
        pivot_col = pc
    return True


class TestFlintSNF:
    def test_identity_p5(self):
        snf, rank = BACKEND.compute_snf(_identity(3), 3, 3, 5)
        assert rank == 3
        assert _is_snf(snf, 3, 3, 3)

    def test_zero_p7(self):
        snf, rank = BACKEND.compute_snf(_zero(3, 3), 3, 3, 7)
        assert rank == 0

    def test_rank1_p5(self):
        M = [[1, 2], [2, 4]]
        snf, rank = BACKEND.compute_snf(M, 2, 2, 5)
        assert rank == 1

    def test_full_rank_3x3_p7(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        snf, rank = BACKEND.compute_snf(M, 3, 3, 7)
        assert rank == 3

    def test_nonsquare_wide_p11(self):
        M = [[1, 0, 0], [0, 1, 0]]
        snf, rank = BACKEND.compute_snf(M, 2, 3, 11)
        assert rank == 2
        assert _is_snf(snf, 2, 3, 2)

    def test_p2(self):
        # [[1,1,0],[0,1,1],[1,0,1]] over F_2: row2 = row0+row1 → rank 2
        M = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        snf, rank = BACKEND.compute_snf(M, 3, 3, 2)
        assert rank == 2


class TestFlintSNFWithTransforms:
    """Flint delegates to pure_python for transforms — just verify correctness."""

    def _verify(self, M, nrows, ncols, p):
        snf, rank, U, V = BACKEND.compute_snf_with_transforms(M, nrows, ncols, p)
        assert _is_snf(snf, nrows, ncols, rank)
        UMV = mat_mul_ff(mat_mul_ff(U, M, p), V, p)
        assert UMV == snf

    def test_identity_p5(self):
        self._verify(_identity(2), 2, 2, 5)

    def test_full_rank_p7(self):
        self._verify([[1, 2, 3], [4, 5, 6], [7, 8, 10]], 3, 3, 7)

    def test_rank_deficient_p5(self):
        self._verify([[1, 2], [2, 4]], 2, 2, 5)


class TestFlintHNF:
    def test_identity_p5(self):
        (H,) = BACKEND.compute_hnf(_identity(3), 3, 3, 5)
        assert H == _identity(3)

    def test_zero_p7(self):
        (H,) = BACKEND.compute_hnf(_zero(2, 2), 2, 2, 7)
        assert H == _zero(2, 2)

    def test_rank1(self):
        M = [[1, 2], [2, 4]]
        (H,) = BACKEND.compute_hnf(M, 2, 2, 5)
        assert _is_rref(H, 2, 2, 5)

    def test_full_rank_3x3_p7(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        (H,) = BACKEND.compute_hnf(M, 3, 3, 7)
        assert H == _identity(3)

    def test_nonsquare_p5(self):
        M = [[1, 0, 2], [0, 1, 3]]
        (H,) = BACKEND.compute_hnf(M, 2, 3, 5)
        assert _is_rref(H, 2, 3, 5)


class TestFlintHNFWithTransform:
    """Flint delegates to pure_python for transforms."""

    def _verify(self, M, nrows, ncols, p):
        H, U = BACKEND.compute_hnf_with_transform(M, nrows, ncols, p)
        assert _is_rref(H, nrows, ncols, p)
        assert mat_mul_ff(U, M, p) == H

    def test_identity_p5(self):
        self._verify(_identity(3), 3, 3, 5)

    def test_full_rank_p7(self):
        self._verify([[1, 2, 3], [4, 5, 6], [7, 8, 10]], 3, 3, 7)

    def test_nonsquare_p11(self):
        self._verify([[1, 2, 3], [4, 5, 6]], 2, 3, 11)


class TestFlintRank:
    def test_full_rank(self):
        assert BACKEND.compute_rank(_identity(3), 3, 3, 7) == 3

    def test_zero(self):
        assert BACKEND.compute_rank(_zero(3, 3), 3, 3, 7) == 0

    def test_rank1(self):
        assert BACKEND.compute_rank([[1, 2], [2, 4]], 2, 2, 5) == 1

    def test_p2_rank2(self):
        # row2 = row0 + row1 over F_2
        M = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        assert BACKEND.compute_rank(M, 3, 3, 2) == 2


class TestFlintVsPurePython:
    """Cross-backend consistency: flint and pure_python must agree."""

    from snforacle.backends.pure_python_ff import PurePythonFFBackend as _PP

    PP = _PP()

    def _check_snf(self, M, nrows, ncols, p):
        snf_f, rank_f = BACKEND.compute_snf(M, nrows, ncols, p)
        snf_p, rank_p = self.PP.compute_snf(M, nrows, ncols, p)
        assert rank_f == rank_p
        assert snf_f == snf_p

    def _check_hnf(self, M, nrows, ncols, p):
        (H_f,) = BACKEND.compute_hnf(M, nrows, ncols, p)
        (H_p,) = self.PP.compute_hnf(M, nrows, ncols, p)
        assert H_f == H_p

    def test_identity_p5(self):
        self._check_snf(_identity(3), 3, 3, 5)
        self._check_hnf(_identity(3), 3, 3, 5)

    def test_rank_deficient(self):
        M = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
        self._check_snf(M, 3, 3, 7)
        self._check_hnf(M, 3, 3, 7)

    def test_full_rank(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        self._check_snf(M, 3, 3, 11)
        self._check_hnf(M, 3, 3, 11)

    def test_wide_matrix(self):
        M = [[1, 0, 2, 3], [0, 1, 4, 2]]
        self._check_snf(M, 2, 4, 7)
        self._check_hnf(M, 2, 4, 7)
