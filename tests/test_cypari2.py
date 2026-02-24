"""Tests for the cypari2 backend via the snforacle JSON interface."""

import pytest

from snforacle import (
    DenseIntMatrix,
    SparseIntMatrix,
    smith_normal_form,
    smith_normal_form_with_transforms,
)
from snforacle.schema import SNFResult, SNFWithTransformsResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Plain Python matrix multiplication (integers only)."""
    m, n, p = len(A), len(A[0]), len(B[0])
    assert len(B) == n
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
        for i in range(m)
    ]


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
# Schema validation tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_dense_wrong_nrows(self):
        with pytest.raises(Exception):
            DenseIntMatrix(format="dense", nrows=3, ncols=2, entries=[[1, 2], [3, 4]])

    def test_dense_wrong_ncols(self):
        with pytest.raises(Exception):
            DenseIntMatrix(format="dense", nrows=2, ncols=3, entries=[[1, 2], [3, 4]])

    def test_sparse_out_of_bounds_row(self):
        with pytest.raises(Exception):
            SparseIntMatrix(
                format="sparse", nrows=2, ncols=2,
                entries=[{"row": 5, "col": 0, "value": 1}],
            )

    def test_sparse_duplicate_entry(self):
        with pytest.raises(Exception):
            SparseIntMatrix(
                format="sparse", nrows=3, ncols=3,
                entries=[
                    {"row": 0, "col": 0, "value": 1},
                    {"row": 0, "col": 0, "value": 2},
                ],
            )

    def test_dict_discriminator_dense(self):
        result = smith_normal_form(_dense_input([[1, 0], [0, 1]]))
        assert isinstance(result, SNFResult)

    def test_dict_discriminator_sparse(self):
        result = smith_normal_form(_sparse_input(2, 2, [(0, 0, 1), (1, 1, 1)]))
        assert isinstance(result, SNFResult)


# ---------------------------------------------------------------------------
# SNF correctness tests
# ---------------------------------------------------------------------------

class TestSNF:
    def test_identity_2x2(self):
        result = smith_normal_form(_dense_input([[1, 0], [0, 1]]))
        assert result.invariant_factors == [1, 1]
        assert result.smith_normal_form.entries == [[1, 0], [0, 1]]

    def test_zero_matrix(self):
        result = smith_normal_form(_dense_input([[0, 0], [0, 0]]))
        # A zero matrix has no nonzero invariant factors.
        assert result.invariant_factors == []
        assert result.smith_normal_form.entries == [[0, 0], [0, 0]]

    def test_3x3_standard(self):
        # M = [[2,4,4],[-6,6,12],[10,-4,-16]]
        # SNF diagonal should be [2, 6, 12]
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form(_dense_input(M))
        assert result.invariant_factors == [2, 6, 12]
        snf = result.smith_normal_form.entries
        assert snf[0][0] == 2 and snf[1][1] == 6 and snf[2][2] == 12
        # off-diagonal zeros
        assert snf[0][1] == snf[0][2] == snf[1][0] == snf[1][2] == snf[2][0] == snf[2][1] == 0

    def test_rank_deficient(self):
        # [[1,2,3],[4,5,6],[7,8,9]] has rank 2; SNF diagonal [1, 3, 0]
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = smith_normal_form(_dense_input(M))
        assert result.invariant_factors == [1, 3]
        snf = result.smith_normal_form.entries
        assert snf[0][0] == 1 and snf[1][1] == 3 and snf[2][2] == 0

    def test_non_square_2x3(self):
        # [[1,2,3],[4,5,6]]; rank 2; SNF diagonal [1, 3]
        M = [[1, 2, 3], [4, 5, 6]]
        result = smith_normal_form(_dense_input(M))
        assert result.invariant_factors == [1, 3]
        snf = result.smith_normal_form.entries
        assert len(snf) == 2 and len(snf[0]) == 3
        assert snf[0][0] == 1 and snf[1][1] == 3

    def test_non_square_3x2(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = smith_normal_form(_dense_input(M))
        assert result.invariant_factors == [1, 3]

    def test_single_element(self):
        result = smith_normal_form(_dense_input([[6]]))
        assert result.invariant_factors == [6]

    def test_sparse_equals_dense(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        dense_result = smith_normal_form(_dense_input(M))
        nonzeros = [
            (r, c, M[r][c]) for r in range(3) for c in range(3) if M[r][c] != 0
        ]
        sparse_result = smith_normal_form(_sparse_input(3, 3, nonzeros))
        assert dense_result.invariant_factors == sparse_result.invariant_factors
        assert dense_result.smith_normal_form.entries == sparse_result.smith_normal_form.entries

    def test_return_type_is_snf_result(self):
        result = smith_normal_form(_dense_input([[3, 0], [0, 5]]))
        assert isinstance(result, SNFResult)
        assert result.smith_normal_form.format == "dense"


# ---------------------------------------------------------------------------
# SNF with transforms tests
# ---------------------------------------------------------------------------

class TestSNFWithTransforms:
    def _verify_factorisation(self, M_entries, result: SNFWithTransformsResult):
        """Check that U @ M @ V == SNF."""
        U = result.left_transform.entries
        V = result.right_transform.entries
        D = result.smith_normal_form.entries
        computed = _mat_mul(_mat_mul(U, M_entries), V)
        assert computed == D, f"U@M@V={computed} != D={D}"

    def test_3x3(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form_with_transforms(_dense_input(M))
        assert result.invariant_factors == [2, 6, 12]
        self._verify_factorisation(M, result)

    def test_2x3_non_square(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = smith_normal_form_with_transforms(_dense_input(M))
        assert result.invariant_factors == [1, 3]
        self._verify_factorisation(M, result)

    def test_3x2_non_square(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = smith_normal_form_with_transforms(_dense_input(M))
        assert result.invariant_factors == [1, 3]
        self._verify_factorisation(M, result)

    def test_sparse_input(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        nonzeros = [(r, c, M[r][c]) for r in range(3) for c in range(3) if M[r][c] != 0]
        result = smith_normal_form_with_transforms(_sparse_input(3, 3, nonzeros))
        assert result.invariant_factors == [2, 6, 12]
        self._verify_factorisation(M, result)

    def test_return_type(self):
        result = smith_normal_form_with_transforms(_dense_input([[1, 2], [3, 4]]))
        assert isinstance(result, SNFWithTransformsResult)
        assert result.left_transform.nrows == 2
        assert result.right_transform.ncols == 2

    def test_json_round_trip(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form_with_transforms(_dense_input(M))
        json_str = result.model_dump_json()
        restored = SNFWithTransformsResult.model_validate_json(json_str)
        assert restored.invariant_factors == result.invariant_factors
        assert restored.smith_normal_form.entries == result.smith_normal_form.entries

    def test_10x10_tridiagonal(self):
        result = smith_normal_form_with_transforms(_dense_input(_M_TRIDIAGONAL_10))
        assert result.invariant_factors == [1, 1, 1, 1, 1, 1, 1, 1, 1, 11]
        self._verify_factorisation(_M_TRIDIAGONAL_10, result)

    def test_10x10_arithmetic_rows(self):
        result = smith_normal_form_with_transforms(_dense_input(_M_ARITHMETIC_10))
        assert result.invariant_factors == [1, 10]
        self._verify_factorisation(_M_ARITHMETIC_10, result)
