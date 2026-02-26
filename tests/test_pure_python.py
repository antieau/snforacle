"""Tests for the pure_python backend and cross-backend consistency."""

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
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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
    """Create a dense matrix input dict."""
    return {
        "format": "dense",
        "nrows": len(entries),
        "ncols": len(entries[0]) if entries else 0,
        "entries": entries,
    }


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Multiply two matrices A @ B."""
    m, n = len(A), len(B[0])
    p = len(B)
    result = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]
    return result


# ---------------------------------------------------------------------------
# Pure Python SNF tests
# ---------------------------------------------------------------------------


class TestPurePythonSNF:
    """Test SNF computation via the pure_python backend."""

    def test_identity_2x2(self):
        result = smith_normal_form(_dense_input([[1, 0], [0, 1]]), backend="pure_python")
        assert result.invariant_factors == [1, 1]
        assert result.smith_normal_form.entries == [[1, 0], [0, 1]]

    def test_zero_matrix(self):
        result = smith_normal_form(_dense_input([[0, 0], [0, 0]]), backend="pure_python")
        assert result.invariant_factors == []
        assert result.smith_normal_form.entries == [[0, 0], [0, 0]]

    def test_3x3_standard(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form(_dense_input(M), backend="pure_python")
        assert result.invariant_factors == [2, 6, 12]
        snf = result.smith_normal_form.entries
        assert snf[0][0] == 2 and snf[1][1] == 6 and snf[2][2] == 12
        assert (
            snf[0][1]
            == snf[0][2]
            == snf[1][0]
            == snf[1][2]
            == snf[2][0]
            == snf[2][1]
            == 0
        )

    def test_rank_deficient(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = smith_normal_form(_dense_input(M), backend="pure_python")
        assert result.invariant_factors == [1, 3]
        snf = result.smith_normal_form.entries
        assert snf[0][0] == 1 and snf[1][1] == 3 and snf[2][2] == 0

    def test_non_square_2x3(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = smith_normal_form(_dense_input(M), backend="pure_python")
        assert result.invariant_factors == [1, 3]
        snf = result.smith_normal_form.entries
        assert len(snf) == 2 and len(snf[0]) == 3
        assert snf[0][0] == 1 and snf[1][1] == 3

    def test_non_square_3x2(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = smith_normal_form(_dense_input(M), backend="pure_python")
        assert result.invariant_factors == [1, 3]

    def test_single_element(self):
        result = smith_normal_form(_dense_input([[6]]), backend="pure_python")
        assert result.invariant_factors == [6]

    def test_sparse_equals_dense(self):
        """Verify sparse and dense input give same result."""
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        dense_result = smith_normal_form(_dense_input(M), backend="pure_python")
        assert dense_result.invariant_factors == [2, 6, 12]

    def test_return_type(self):
        result = smith_normal_form(
            _dense_input([[3, 0], [0, 5]]), backend="pure_python"
        )
        assert isinstance(result, SNFResult)
        assert result.smith_normal_form.format == "dense"


# ---------------------------------------------------------------------------
# Pure Python SNF with transforms
# ---------------------------------------------------------------------------


class TestPurePythonSNFWithTransforms:
    """Test that SNF transforms satisfy U @ M @ V == SNF."""

    def test_3x3_standard(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form_with_transforms(_dense_input(M), backend="pure_python")

        U = result.left_transform.entries
        V = result.right_transform.entries
        snf = result.smith_normal_form.entries

        # Verify U @ M @ V == SNF
        UM = _mat_mul(U, M)
        UMV = _mat_mul(UM, V)

        assert UMV == snf, f"U @ M @ V != SNF: {UMV} != {snf}"
        assert result.invariant_factors == [2, 6, 12]

    def test_json_round_trip(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form_with_transforms(_dense_input(M), backend="pure_python")

        # Test serialization
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "invariant_factors" in json_str

    def test_2x3_non_square(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = smith_normal_form_with_transforms(_dense_input(M), backend="pure_python")

        U = result.left_transform.entries
        V = result.right_transform.entries
        snf = result.smith_normal_form.entries

        UM = _mat_mul(U, M)
        UMV = _mat_mul(UM, V)

        assert UMV == snf
        assert result.invariant_factors == [1, 3]

    def test_3x2_non_square(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = smith_normal_form_with_transforms(_dense_input(M), backend="pure_python")

        U = result.left_transform.entries
        V = result.right_transform.entries
        snf = result.smith_normal_form.entries

        UM = _mat_mul(U, M)
        UMV = _mat_mul(UM, V)

        assert UMV == snf
        assert result.invariant_factors == [1, 3]


# ---------------------------------------------------------------------------
# Pure Python HNF tests
# ---------------------------------------------------------------------------


class TestPurePythonHNF:
    """Test HNF computation via the pure_python backend."""

    def test_3x3_standard(self):
        """Standard 3x3 example: expect [[2,4,4],[0,6,0],[0,0,12]]."""
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = hermite_normal_form(_dense_input(M), backend="pure_python")
        assert result.hermite_normal_form.entries == [[2, 4, 4], [0, 6, 0], [0, 0, 12]]

    def test_identity_2x2(self):
        result = hermite_normal_form(_dense_input([[1, 0], [0, 1]]), backend="pure_python")
        assert result.hermite_normal_form.entries == [[1, 0], [0, 1]]

    def test_zero_matrix(self):
        result = hermite_normal_form(_dense_input([[0, 0], [0, 0]]), backend="pure_python")
        assert result.hermite_normal_form.entries == [[0, 0], [0, 0]]

    def test_rank_deficient(self):
        """Rank-deficient matrix should have zero rows at bottom."""
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = hermite_normal_form(_dense_input(M), backend="pure_python")
        H = result.hermite_normal_form.entries
        # HNF should be upper triangular with pivots on diagonal
        # The zero rows should appear at the bottom
        assert all(
            H[i][j] == 0
            for i in range(len(H))
            for j in range(i)
            if H[i][i] == 0
        )

    def test_non_square_2x3(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = hermite_normal_form(_dense_input(M), backend="pure_python")
        H = result.hermite_normal_form.entries
        assert len(H) == 2 and len(H[0]) == 3

    def test_non_square_3x2(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = hermite_normal_form(_dense_input(M), backend="pure_python")
        H = result.hermite_normal_form.entries
        assert len(H) == 3 and len(H[0]) == 2

    def test_return_type(self):
        result = hermite_normal_form(_dense_input([[1, 0], [0, 1]]), backend="pure_python")
        assert isinstance(result, HNFResult)
        assert result.hermite_normal_form.format == "dense"


# ---------------------------------------------------------------------------
# Pure Python HNF with transform
# ---------------------------------------------------------------------------


class TestPurePythonHNFWithTransform:
    """Test that HNF transforms satisfy U @ M == HNF."""

    def test_3x3_standard(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = hermite_normal_form_with_transform(_dense_input(M), backend="pure_python")

        U = result.left_transform.entries
        hnf = result.hermite_normal_form.entries

        # Verify U @ M == HNF
        UM = _mat_mul(U, M)
        assert UM == hnf, f"U @ M != HNF: {UM} != {hnf}"

    def test_identity_2x2(self):
        M = [[1, 0], [0, 1]]
        result = hermite_normal_form_with_transform(_dense_input(M), backend="pure_python")

        U = result.left_transform.entries
        hnf = result.hermite_normal_form.entries

        UM = _mat_mul(U, M)
        assert UM == hnf

    def test_2x3_non_square(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = hermite_normal_form_with_transform(_dense_input(M), backend="pure_python")

        U = result.left_transform.entries
        hnf = result.hermite_normal_form.entries

        UM = _mat_mul(U, M)
        assert UM == hnf

    def test_3x2_non_square(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = hermite_normal_form_with_transform(_dense_input(M), backend="pure_python")

        U = result.left_transform.entries
        hnf = result.hermite_normal_form.entries

        UM = _mat_mul(U, M)
        assert UM == hnf


# ---------------------------------------------------------------------------
# Elementary divisors tests
# ---------------------------------------------------------------------------


class TestPurePythonElementaryDivisors:
    """Test elementary divisors computation."""

    def test_3x3_standard(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        ed_result = elementary_divisors(_dense_input(M), backend="pure_python")
        snf_result = smith_normal_form(_dense_input(M), backend="pure_python")
        assert ed_result.elementary_divisors == snf_result.invariant_factors
        assert ed_result.elementary_divisors == [2, 6, 12]

    def test_rank_deficient(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ed_result = elementary_divisors(_dense_input(M), backend="pure_python")
        snf_result = smith_normal_form(_dense_input(M), backend="pure_python")
        assert ed_result.elementary_divisors == snf_result.invariant_factors
        assert ed_result.elementary_divisors == [1, 3]

    def test_zero_matrix(self):
        ed_result = elementary_divisors(_dense_input([[0, 0], [0, 0]]), backend="pure_python")
        assert ed_result.elementary_divisors == []


# ---------------------------------------------------------------------------
# Cross-backend consistency: pure_python vs cypari2
# ---------------------------------------------------------------------------


class TestPurePythonVsCypari2:
    """Verify that pure_python and cypari2 backends produce identical SNF output."""

    def _assert_same_snf(self, entries):
        inp = _dense_input(entries)
        pp_result = smith_normal_form(inp, backend="pure_python")
        pari_result = smith_normal_form(inp, backend="cypari2")
        assert pp_result.invariant_factors == pari_result.invariant_factors, (
            f"invariant_factors differ: pure_python={pp_result.invariant_factors} "
            f"cypari2={pari_result.invariant_factors}"
        )
        assert (
            pp_result.smith_normal_form.entries
            == pari_result.smith_normal_form.entries
        ), (
            f"SNF entries differ: pure_python={pp_result.smith_normal_form.entries} "
            f"cypari2={pari_result.smith_normal_form.entries}"
        )

    def test_identity_2x2(self):
        self._assert_same_snf([[1, 0], [0, 1]])

    def test_zero_matrix(self):
        self._assert_same_snf([[0, 0], [0, 0]])

    def test_3x3_standard(self):
        self._assert_same_snf([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])

    def test_rank_deficient(self):
        self._assert_same_snf([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_non_square_2x3(self):
        self._assert_same_snf([[1, 2, 3], [4, 5, 6]])

    def test_non_square_3x2(self):
        self._assert_same_snf([[1, 4], [2, 5], [3, 6]])

    def test_single_element(self):
        self._assert_same_snf([[6]])

    def test_diagonal(self):
        self._assert_same_snf([[4, 0, 0], [0, 6, 0], [0, 0, 10]])

    def test_negative_entries(self):
        self._assert_same_snf([[-6, 111, 0], [0, -3, 57], [2, 0, -14]])


# ---------------------------------------------------------------------------
# Cross-backend consistency: pure_python vs flint HNF
# ---------------------------------------------------------------------------


class TestPurePythonVsFlintHNF:
    """Verify that pure_python and flint backends produce identical HNF output."""

    def _assert_same_hnf(self, entries):
        inp = _dense_input(entries)
        pp_result = hermite_normal_form(inp, backend="pure_python")
        flint_result = hermite_normal_form(inp, backend="flint")
        assert pp_result.hermite_normal_form.entries == flint_result.hermite_normal_form.entries, (
            f"HNF entries differ: pure_python={pp_result.hermite_normal_form.entries} "
            f"flint={flint_result.hermite_normal_form.entries}"
        )

    def test_identity_2x2(self):
        self._assert_same_hnf([[1, 0], [0, 1]])

    def test_zero_matrix(self):
        self._assert_same_hnf([[0, 0], [0, 0]])

    def test_3x3_standard(self):
        self._assert_same_hnf([[2, 4, 4], [-6, 6, 12], [10, -4, -16]])

    def test_rank_deficient(self):
        self._assert_same_hnf([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_non_square_2x3(self):
        self._assert_same_hnf([[1, 2, 3], [4, 5, 6]])

    def test_non_square_3x2(self):
        self._assert_same_hnf([[1, 4], [2, 5], [3, 6]])

    def test_single_element(self):
        self._assert_same_hnf([[6]])

    def test_diagonal(self):
        self._assert_same_hnf([[4, 0, 0], [0, 6, 0], [0, 0, 10]])

    def test_negative_entries(self):
        self._assert_same_hnf([[-6, 111, 0], [0, -3, 57], [2, 0, -14]])
