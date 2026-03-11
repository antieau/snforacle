"""Tests for the pure Python finite-field backend."""

from __future__ import annotations

import pytest

from _mathelpers import assert_invertible_ff
from snforacle.backends.pure_python_ff import (
    PurePythonFFBackend,
    _hnf_with_transform,
    _snf_with_transforms,
    mat_mul_ff,
)

BACKEND = PurePythonFFBackend()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _zero(nrows: int, ncols: int) -> list[list[int]]:
    return [[0] * ncols for _ in range(nrows)]


def _mat_eq(A: list[list[int]], B: list[list[int]]) -> bool:
    return A == B


def _is_snf(snf: list[list[int]], nrows: int, ncols: int, rank: int) -> bool:
    """Check that snf is diag(1,...,1,0,...,0) with *rank* leading 1s."""
    for i in range(nrows):
        for j in range(ncols):
            expected = 1 if (i == j and i < rank) else 0
            if snf[i][j] != expected:
                return False
    return True


def _is_rref(H: list[list[int]], nrows: int, ncols: int, p: int) -> bool:
    """Check that H is in RREF over F_p."""
    pivot_col = -1
    for i in range(nrows):
        # Find pivot in row i
        pc = None
        for j in range(ncols):
            if H[i][j] != 0:
                pc = j
                break
        if pc is None:
            # Zero row — all subsequent rows must also be zero
            for ii in range(i + 1, nrows):
                if any(H[ii][j] != 0 for j in range(ncols)):
                    return False
            break
        if pc <= pivot_col:
            return False  # pivot columns must be strictly increasing
        if H[i][pc] != 1:
            return False  # pivot must be 1
        # All other entries in pivot column must be 0
        for ii in range(nrows):
            if ii != i and H[ii][pc] != 0:
                return False
        pivot_col = pc
    return True


# ---------------------------------------------------------------------------
# SNF tests
# ---------------------------------------------------------------------------

class TestSNF:
    def test_identity_2x2_p5(self):
        M = _identity(2)
        snf, rank = BACKEND.compute_snf(M, 2, 2, 5)
        assert rank == 2
        assert _is_snf(snf, 2, 2, 2)

    def test_zero_3x3_p7(self):
        M = _zero(3, 3)
        snf, rank = BACKEND.compute_snf(M, 3, 3, 7)
        assert rank == 0
        assert _is_snf(snf, 3, 3, 0)

    def test_rank1_p5(self):
        M = [[1, 2], [2, 4]]  # rank 1 over any field (row 2 = 2*row 1)
        snf, rank = BACKEND.compute_snf(M, 2, 2, 5)
        assert rank == 1
        assert _is_snf(snf, 2, 2, 1)

    def test_rank1_p2(self):
        M = [[1, 0], [1, 0]]
        snf, rank = BACKEND.compute_snf(M, 2, 2, 2)
        assert rank == 1
        assert _is_snf(snf, 2, 2, 1)

    def test_nonsquare_2x3_p3(self):
        M = [[1, 0, 0], [0, 1, 0]]
        snf, rank = BACKEND.compute_snf(M, 2, 3, 3)
        assert rank == 2
        assert _is_snf(snf, 2, 3, 2)

    def test_nonsquare_3x2_p7(self):
        M = [[1, 0], [0, 1], [0, 0]]
        snf, rank = BACKEND.compute_snf(M, 3, 2, 7)
        assert rank == 2
        assert _is_snf(snf, 3, 2, 2)

    def test_full_rank_3x3_p11(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        snf, rank = BACKEND.compute_snf(M, 3, 3, 11)
        assert rank == 3
        assert _is_snf(snf, 3, 3, 3)

    def test_singular_3x3_p5(self):
        # rows sum to zero mod 5
        M = [[1, 2, 3], [4, 0, 1], [0, 3, 1]]
        # det = 1*(0-3) - 2*(4-0) + 3*(12-0) = -3 - 8 + 36 = 25 ≡ 0 mod 5
        snf, rank = BACKEND.compute_snf(M, 3, 3, 5)
        assert rank == 2
        assert _is_snf(snf, 3, 3, 2)

    def test_entries_reduced_mod_p(self):
        # entries larger than p are valid if they came in; backend should still work
        # but actually ff_schema validates [0, p-1]. Test backend directly.
        M = [[3, 6], [9, 12]]  # mod 3 → [[0,0],[0,0]]
        snf, rank = BACKEND.compute_snf(M, 2, 2, 3)
        assert rank == 0


# ---------------------------------------------------------------------------
# SNF with transforms tests
# ---------------------------------------------------------------------------

class TestSNFWithTransforms:
    def _verify(self, M, nrows, ncols, p):
        snf, rank, U, V = _snf_with_transforms(M, nrows, ncols, p)
        assert _is_snf(snf, nrows, ncols, rank)
        # Check U @ M @ V = snf
        UM = mat_mul_ff(U, M, p)
        UMV = mat_mul_ff(UM, V, p)
        assert UMV == snf, f"U @ M @ V ≠ snf\nU={U}\nM={M}\nV={V}\nUMV={UMV}\nsnf={snf}"
        assert_invertible_ff(U, p, "U")
        assert_invertible_ff(V, p, "V")
        return snf, rank, U, V

    def test_identity_2x2_p5(self):
        self._verify(_identity(2), 2, 2, 5)

    def test_zero_2x2_p7(self):
        self._verify(_zero(2, 2), 2, 2, 7)

    def test_rank1_p5(self):
        self._verify([[1, 2], [2, 4]], 2, 2, 5)

    def test_full_rank_3x3_p7(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        self._verify(M, 3, 3, 7)

    def test_nonsquare_2x3_p11(self):
        M = [[1, 2, 3], [4, 5, 6]]
        self._verify(M, 2, 3, 11)

    def test_nonsquare_3x2_p5(self):
        M = [[1, 2], [3, 4], [5, 6]]
        self._verify(M, 3, 2, 5)

    def test_rank_deficient_3x3_p13(self):
        M = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]  # rank 1
        self._verify(M, 3, 3, 13)

    def test_p2_binary(self):
        M = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        self._verify(M, 3, 3, 2)

    def test_1x1_nonzero_p7(self):
        self._verify([[3]], 1, 1, 7)

    def test_1x1_zero_p7(self):
        self._verify([[0]], 1, 1, 7)


# ---------------------------------------------------------------------------
# HNF tests
# ---------------------------------------------------------------------------

class TestHNF:
    def test_identity_2x2_p5(self):
        M = _identity(2)
        (H,) = BACKEND.compute_hnf(M, 2, 2, 5)
        assert _is_rref(H, 2, 2, 5)
        assert H == [[1, 0], [0, 1]]

    def test_zero_2x2_p5(self):
        M = _zero(2, 2)
        (H,) = BACKEND.compute_hnf(M, 2, 2, 5)
        assert H == [[0, 0], [0, 0]]

    def test_rank1_p5(self):
        M = [[1, 2], [2, 4]]
        (H,) = BACKEND.compute_hnf(M, 2, 2, 5)
        assert _is_rref(H, 2, 2, 5)
        assert H[1] == [0, 0]  # second row is zero

    def test_full_rank_3x3_p7(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        (H,) = BACKEND.compute_hnf(M, 3, 3, 7)
        assert _is_rref(H, 3, 3, 7)
        assert H == _identity(3)

    def test_nonsquare_2x4_p5(self):
        M = [[1, 0, 2, 3], [0, 1, 4, 1]]
        (H,) = BACKEND.compute_hnf(M, 2, 4, 5)
        assert _is_rref(H, 2, 4, 5)

    def test_consistency_with_snf_rank(self):
        M = [[2, 3, 1], [4, 1, 5], [6, 4, 6]]
        (H,) = BACKEND.compute_hnf(M, 3, 3, 7)
        snf, rank = BACKEND.compute_snf(M, 3, 3, 7)
        # number of nonzero rows in H = rank
        nonzero_rows = sum(1 for row in H if any(v != 0 for v in row))
        assert nonzero_rows == rank


# ---------------------------------------------------------------------------
# HNF with transform tests
# ---------------------------------------------------------------------------

class TestHNFWithTransform:
    def _verify(self, M, nrows, ncols, p):
        H, U = _hnf_with_transform(M, nrows, ncols, p)
        assert _is_rref(H, nrows, ncols, p)
        UH = mat_mul_ff(U, M, p)
        assert UH == H, f"U @ M ≠ H\nU={U}\nM={M}\nUH={UH}\nH={H}"
        assert_invertible_ff(U, p, "U")
        return H, U

    def test_identity_p5(self):
        self._verify(_identity(3), 3, 3, 5)

    def test_zero_p7(self):
        self._verify(_zero(3, 3), 3, 3, 7)

    def test_rank1_p5(self):
        self._verify([[1, 2], [2, 4]], 2, 2, 5)

    def test_full_rank_3x3_p11(self):
        self._verify([[1, 2, 3], [4, 5, 6], [7, 8, 10]], 3, 3, 11)

    def test_nonsquare_2x3_p7(self):
        self._verify([[1, 2, 3], [4, 5, 6]], 2, 3, 7)

    def test_nonsquare_3x2_p5(self):
        self._verify([[1, 2], [3, 4], [5, 6]], 3, 2, 5)

    def test_p2(self):
        self._verify([[1, 1, 0], [1, 0, 1], [0, 1, 1]], 3, 3, 2)


# ---------------------------------------------------------------------------
# Rank tests
# ---------------------------------------------------------------------------

class TestRank:
    def test_full_rank(self):
        assert BACKEND.compute_rank(_identity(3), 3, 3, 7) == 3

    def test_zero(self):
        assert BACKEND.compute_rank(_zero(3, 3), 3, 3, 7) == 0

    def test_rank1(self):
        assert BACKEND.compute_rank([[1, 2], [2, 4]], 2, 2, 5) == 1

    def test_rank2_of_3(self):
        M = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        assert BACKEND.compute_rank(M, 3, 3, 5) == 2

    def test_nonsquare_wide(self):
        M = [[1, 0, 0, 0], [0, 1, 0, 0]]
        assert BACKEND.compute_rank(M, 2, 4, 7) == 2

    def test_nonsquare_tall(self):
        M = [[1, 0], [0, 1], [0, 0], [0, 0]]
        assert BACKEND.compute_rank(M, 4, 2, 7) == 2


# ---------------------------------------------------------------------------
# Public API (ff_interface) tests
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_snf_basic(self):
        from snforacle import ff_smith_normal_form
        result = ff_smith_normal_form(
            {"format": "dense_ff", "nrows": 2, "ncols": 2, "p": 5,
             "entries": [[1, 2], [3, 4]]}
        )
        assert result.rank == 2
        assert result.smith_normal_form.entries == [[1, 0], [0, 1]]

    def test_snf_zero_matrix(self):
        from snforacle import ff_smith_normal_form
        result = ff_smith_normal_form(
            {"format": "dense_ff", "nrows": 2, "ncols": 2, "p": 5,
             "entries": [[0, 0], [0, 0]]}
        )
        assert result.rank == 0

    def test_snf_empty_rows(self):
        from snforacle import ff_smith_normal_form
        result = ff_smith_normal_form(
            {"format": "dense_ff", "nrows": 0, "ncols": 3, "p": 5, "entries": []}
        )
        assert result.rank == 0

    def test_snf_with_transforms_basic(self):
        from snforacle import ff_smith_normal_form_with_transforms
        M = [[1, 2], [3, 4]]
        result = ff_smith_normal_form_with_transforms(
            {"format": "dense_ff", "nrows": 2, "ncols": 2, "p": 5, "entries": M},
            backend="pure_python",
        )
        assert result.rank == 2
        U = result.left_transform.entries
        V = result.right_transform.entries
        UMV = mat_mul_ff(mat_mul_ff(U, M, 5), V, 5)
        assert UMV == [[1, 0], [0, 1]]
        assert_invertible_ff(U, 5, "U")
        assert_invertible_ff(V, 5, "V")

    def test_hnf_basic(self):
        from snforacle import ff_hermite_normal_form
        M = [[2, 4, 6], [1, 2, 3]]
        result = ff_hermite_normal_form(
            {"format": "dense_ff", "nrows": 2, "ncols": 3, "p": 7, "entries": M}
        )
        H = result.hermite_normal_form.entries
        assert _is_rref(H, 2, 3, 7)

    def test_hnf_with_transform_basic(self):
        from snforacle import ff_hermite_normal_form_with_transform
        M = [[1, 2, 3], [4, 5, 6]]
        result = ff_hermite_normal_form_with_transform(
            {"format": "dense_ff", "nrows": 2, "ncols": 3, "p": 7, "entries": M}
        )
        H = result.hermite_normal_form.entries
        U = result.left_transform.entries
        assert mat_mul_ff(U, M, 7) == H
        assert_invertible_ff(U, 7, "U")

    def test_rank_basic(self):
        from snforacle import ff_rank
        result = ff_rank(
            {"format": "dense_ff", "nrows": 3, "ncols": 3, "p": 7,
             "entries": [[1, 0, 0], [0, 1, 0], [0, 0, 0]]}
        )
        assert result.rank == 2

    def test_sparse_input(self):
        from snforacle import ff_rank
        result = ff_rank({
            "format": "sparse_ff",
            "nrows": 3,
            "ncols": 3,
            "p": 7,
            "entries": [
                {"row": 0, "col": 0, "value": 1},
                {"row": 1, "col": 1, "value": 1},
            ],
        })
        assert result.rank == 2

    def test_backend_explicit_pure_python(self):
        from snforacle import ff_rank
        result = ff_rank(
            {"format": "dense_ff", "nrows": 2, "ncols": 2, "p": 5,
             "entries": [[1, 0], [0, 1]]},
            backend="pure_python",
        )
        assert result.rank == 2
