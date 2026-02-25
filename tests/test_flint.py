"""Tests for the flint backend and cross-backend consistency with cypari2."""

import pytest

from snforacle import (
    elementary_divisors,
    hermite_normal_form,
    hermite_normal_form_with_transform,
    smith_normal_form,
    smith_normal_form_with_transforms,
)
from snforacle.schema import HNFResult, SNFResult


# ---------------------------------------------------------------------------
# Shared 10x10 test matrices
# ---------------------------------------------------------------------------

# Tridiagonal matrix: 2 on the main diagonal, 1 on the two adjacent diagonals.
# det = 11 (prime), so SNF = diag(1,1,...,1,11).
_M_TRIDIAGONAL_10 = [
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 2, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 2, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 2, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
]

# Arithmetic rows: row i has entries 10i+1, 10i+2, ..., 10i+10.
# Rank 2, SNF = diag(1, 10, 0, ..., 0).
_M_ARITHMETIC_10 = [
    [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
    [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
    [71, 72, 73, 74, 75, 76, 77, 78, 79, 80],
    [81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    [91, 92, 93, 94, 95, 96, 97, 98, 99, 100],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense_input(entries: list[list[int]]) -> dict:
    return {
        "format": "dense",
        "nrows": len(entries),
        "ncols": len(entries[0]),
        "entries": entries,
    }


def _sparse_input(nrows: int, ncols: int, nonzeros: list[tuple[int, int, int]]) -> dict:
    return {
        "format": "sparse",
        "nrows": nrows,
        "ncols": ncols,
        "entries": [{"row": r, "col": c, "value": v} for r, c, v in nonzeros],
    }


# ---------------------------------------------------------------------------
# Flint backend SNF correctness tests
# ---------------------------------------------------------------------------

class TestFlintSNF:
    def test_identity_2x2(self):
        result = smith_normal_form(_dense_input([[1, 0], [0, 1]]), backend="flint")
        assert result.invariant_factors == [1, 1]
        assert result.smith_normal_form.entries == [[1, 0], [0, 1]]

    def test_zero_matrix(self):
        result = smith_normal_form(_dense_input([[0, 0], [0, 0]]), backend="flint")
        assert result.invariant_factors == []
        assert result.smith_normal_form.entries == [[0, 0], [0, 0]]

    def test_3x3_standard(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form(_dense_input(M), backend="flint")
        assert result.invariant_factors == [2, 6, 12]
        snf = result.smith_normal_form.entries
        assert snf[0][0] == 2 and snf[1][1] == 6 and snf[2][2] == 12
        assert snf[0][1] == snf[0][2] == snf[1][0] == snf[1][2] == snf[2][0] == snf[2][1] == 0

    def test_rank_deficient(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = smith_normal_form(_dense_input(M), backend="flint")
        assert result.invariant_factors == [1, 3]
        snf = result.smith_normal_form.entries
        assert snf[0][0] == 1 and snf[1][1] == 3 and snf[2][2] == 0

    def test_non_square_2x3(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = smith_normal_form(_dense_input(M), backend="flint")
        assert result.invariant_factors == [1, 3]
        snf = result.smith_normal_form.entries
        assert len(snf) == 2 and len(snf[0]) == 3
        assert snf[0][0] == 1 and snf[1][1] == 3

    def test_non_square_3x2(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = smith_normal_form(_dense_input(M), backend="flint")
        assert result.invariant_factors == [1, 3]

    def test_single_element(self):
        result = smith_normal_form(_dense_input([[6]]), backend="flint")
        assert result.invariant_factors == [6]

    def test_sparse_equals_dense(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        dense_result = smith_normal_form(_dense_input(M), backend="flint")
        nonzeros = [
            (r, c, M[r][c]) for r in range(3) for c in range(3) if M[r][c] != 0
        ]
        sparse_result = smith_normal_form(_sparse_input(3, 3, nonzeros), backend="flint")
        assert dense_result.invariant_factors == sparse_result.invariant_factors
        assert dense_result.smith_normal_form.entries == sparse_result.smith_normal_form.entries

    def test_return_type(self):
        result = smith_normal_form(_dense_input([[3, 0], [0, 5]]), backend="flint")
        assert isinstance(result, SNFResult)
        assert result.smith_normal_form.format == "dense"

    def test_transforms_not_implemented(self):
        from snforacle.backends.flint import FlintBackend
        backend = FlintBackend()
        with pytest.raises(NotImplementedError):
            backend.compute_snf_with_transforms([[1, 2], [3, 4]], 2, 2)

    def test_transforms_via_interface_not_implemented(self):
        with pytest.raises(NotImplementedError):
            smith_normal_form_with_transforms(_dense_input([[1, 2], [3, 4]]), backend="flint")


# ---------------------------------------------------------------------------
# Cross-backend consistency: flint vs cypari2
# ---------------------------------------------------------------------------

class TestFlintVsCypari2:
    """Verify that the flint and cypari2 backends produce identical output."""

    def _assert_same(self, entries):
        inp = _dense_input(entries)
        flint_result = smith_normal_form(inp, backend="flint")
        pari_result = smith_normal_form(inp, backend="cypari2")
        assert flint_result.invariant_factors == pari_result.invariant_factors, (
            f"invariant_factors differ: flint={flint_result.invariant_factors} "
            f"cypari2={pari_result.invariant_factors}"
        )
        assert flint_result.smith_normal_form.entries == pari_result.smith_normal_form.entries, (
            f"SNF entries differ: flint={flint_result.smith_normal_form.entries} "
            f"cypari2={pari_result.smith_normal_form.entries}"
        )

    def test_identity_2x2(self):
        self._assert_same([[1, 0], [0, 1]])

    def test_zero_matrix(self):
        self._assert_same([[0, 0], [0, 0]])

    def test_3x3_standard(self):
        self._assert_same([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])

    def test_rank_deficient(self):
        self._assert_same([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_non_square_2x3(self):
        self._assert_same([[1, 2, 3], [4, 5, 6]])

    def test_non_square_3x2(self):
        self._assert_same([[1, 4], [2, 5], [3, 6]])

    def test_single_element(self):
        self._assert_same([[6]])

    def test_diagonal(self):
        self._assert_same([[4, 0, 0], [0, 6, 0], [0, 0, 10]])

    def test_negative_entries(self):
        self._assert_same([[-6, 111, 0], [0, -3, 57], [2, 0, -14]])

    def test_10x10_tridiagonal(self):
        self._assert_same(_M_TRIDIAGONAL_10)

    def test_10x10_arithmetic_rows(self):
        self._assert_same(_M_ARITHMETIC_10)


# ---------------------------------------------------------------------------
# Flint HNF tests
# ---------------------------------------------------------------------------

class TestFlintHNF:
    """Test HNF computation via the flint backend."""

    def test_3x3_standard(self):
        """Standard 3x3 example: expect [[2,4,4],[0,6,0],[0,0,12]]."""
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = hermite_normal_form(_dense_input(M), backend="flint")
        assert result.hermite_normal_form.entries == [[2, 4, 4], [0, 6, 0], [0, 0, 12]]

    def test_identity_2x2(self):
        result = hermite_normal_form(_dense_input([[1, 0], [0, 1]]), backend="flint")
        assert result.hermite_normal_form.entries == [[1, 0], [0, 1]]

    def test_zero_matrix(self):
        result = hermite_normal_form(_dense_input([[0, 0], [0, 0]]), backend="flint")
        assert result.hermite_normal_form.entries == [[0, 0], [0, 0]]

    def test_rank_deficient(self):
        """Rank-deficient matrix should have zero rows at bottom."""
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = hermite_normal_form(_dense_input(M), backend="flint")
        H = result.hermite_normal_form.entries
        # HNF should be upper triangular with pivots on diagonal
        # The zero rows should appear at the bottom
        assert all(H[i][j] == 0 for i in range(len(H)) for j in range(i) if H[i][i] == 0)

    def test_non_square_2x3(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = hermite_normal_form(_dense_input(M), backend="flint")
        H = result.hermite_normal_form.entries
        assert len(H) == 2 and len(H[0]) == 3

    def test_non_square_3x2(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = hermite_normal_form(_dense_input(M), backend="flint")
        H = result.hermite_normal_form.entries
        assert len(H) == 3 and len(H[0]) == 2

    def test_return_type(self):
        result = hermite_normal_form(_dense_input([[1, 0], [0, 1]]), backend="flint")
        assert isinstance(result, HNFResult)
        assert result.hermite_normal_form.format == "dense"

    def test_hnf_with_transform_not_implemented(self):
        """Flint 0.8.0 does not expose hnf_transform."""
        M = _dense_input([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])
        with pytest.raises(NotImplementedError):
            hermite_normal_form_with_transform(M, backend="flint")

    def test_elementary_divisors_matches_snf(self):
        """Elementary divisors should match SNF invariant factors."""
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        ed_result = elementary_divisors(_dense_input(M), backend="flint")
        snf_result = smith_normal_form(_dense_input(M), backend="flint")
        assert ed_result.elementary_divisors == snf_result.invariant_factors
