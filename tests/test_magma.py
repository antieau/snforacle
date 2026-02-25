"""Tests for the MAGMA backend and cross-backend consistency with cypari2."""

import shutil

import pytest

from snforacle import smith_normal_form, smith_normal_form_with_transforms
from snforacle.schema import SNFResult, SNFWithTransformsResult

pytestmark = pytest.mark.skipif(
    shutil.which("magma") is None,
    reason="magma binary not found on PATH",
)

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


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Plain Python matrix multiplication (integers only)."""
    m, n, p = len(A), len(A[0]), len(B[0])
    assert len(B) == n
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
        for i in range(m)
    ]


# ---------------------------------------------------------------------------
# MAGMA backend SNF correctness tests
# ---------------------------------------------------------------------------

class TestMagmaSNF:
    def test_identity_2x2(self):
        result = smith_normal_form(_dense_input([[1, 0], [0, 1]]), backend="magma")
        assert result.invariant_factors == [1, 1]
        assert result.smith_normal_form.entries == [[1, 0], [0, 1]]

    def test_zero_matrix(self):
        result = smith_normal_form(_dense_input([[0, 0], [0, 0]]), backend="magma")
        assert result.invariant_factors == []
        assert result.smith_normal_form.entries == [[0, 0], [0, 0]]

    def test_3x3_standard(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form(_dense_input(M), backend="magma")
        assert result.invariant_factors == [2, 6, 12]
        snf = result.smith_normal_form.entries
        assert snf[0][0] == 2 and snf[1][1] == 6 and snf[2][2] == 12
        assert snf[0][1] == snf[0][2] == snf[1][0] == snf[1][2] == snf[2][0] == snf[2][1] == 0

    def test_rank_deficient(self):
        M = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = smith_normal_form(_dense_input(M), backend="magma")
        assert result.invariant_factors == [1, 3]
        snf = result.smith_normal_form.entries
        assert snf[0][0] == 1 and snf[1][1] == 3 and snf[2][2] == 0

    def test_non_square_2x3(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = smith_normal_form(_dense_input(M), backend="magma")
        assert result.invariant_factors == [1, 3]
        snf = result.smith_normal_form.entries
        assert len(snf) == 2 and len(snf[0]) == 3
        assert snf[0][0] == 1 and snf[1][1] == 3

    def test_non_square_3x2(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = smith_normal_form(_dense_input(M), backend="magma")
        assert result.invariant_factors == [1, 3]

    def test_single_element(self):
        result = smith_normal_form(_dense_input([[6]]), backend="magma")
        assert result.invariant_factors == [6]

    def test_sparse_equals_dense(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        dense_result = smith_normal_form(_dense_input(M), backend="magma")
        nonzeros = [
            (r, c, M[r][c]) for r in range(3) for c in range(3) if M[r][c] != 0
        ]
        sparse_result = smith_normal_form(_sparse_input(3, 3, nonzeros), backend="magma")
        assert dense_result.invariant_factors == sparse_result.invariant_factors
        assert dense_result.smith_normal_form.entries == sparse_result.smith_normal_form.entries

    def test_return_type(self):
        result = smith_normal_form(_dense_input([[3, 0], [0, 5]]), backend="magma")
        assert isinstance(result, SNFResult)
        assert result.smith_normal_form.format == "dense"

    def test_10x10_tridiagonal(self):
        result = smith_normal_form(_dense_input(_M_TRIDIAGONAL_10), backend="magma")
        assert result.invariant_factors == [1, 1, 1, 1, 1, 1, 1, 1, 1, 11]

    def test_10x10_arithmetic_rows(self):
        result = smith_normal_form(_dense_input(_M_ARITHMETIC_10), backend="magma")
        assert result.invariant_factors == [1, 10]


# ---------------------------------------------------------------------------
# MAGMA backend SNF with transforms tests
# ---------------------------------------------------------------------------

class TestMagmaSNFWithTransforms:
    def _verify_factorisation(self, M_entries, result: SNFWithTransformsResult):
        """Check that U @ M @ V == SNF."""
        U = result.left_transform.entries
        V = result.right_transform.entries
        D = result.smith_normal_form.entries
        computed = _mat_mul(_mat_mul(U, M_entries), V)
        assert computed == D, f"U@M@V={computed} != D={D}"

    def test_3x3(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form_with_transforms(_dense_input(M), backend="magma")
        assert result.invariant_factors == [2, 6, 12]
        self._verify_factorisation(M, result)

    def test_2x3_non_square(self):
        M = [[1, 2, 3], [4, 5, 6]]
        result = smith_normal_form_with_transforms(_dense_input(M), backend="magma")
        assert result.invariant_factors == [1, 3]
        self._verify_factorisation(M, result)

    def test_3x2_non_square(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = smith_normal_form_with_transforms(_dense_input(M), backend="magma")
        assert result.invariant_factors == [1, 3]
        self._verify_factorisation(M, result)

    def test_return_type(self):
        result = smith_normal_form_with_transforms(
            _dense_input([[1, 2], [3, 4]]), backend="magma"
        )
        assert isinstance(result, SNFWithTransformsResult)
        assert result.left_transform.nrows == 2
        assert result.right_transform.ncols == 2

    def test_10x10_tridiagonal(self):
        result = smith_normal_form_with_transforms(
            _dense_input(_M_TRIDIAGONAL_10), backend="magma"
        )
        assert result.invariant_factors == [1, 1, 1, 1, 1, 1, 1, 1, 1, 11]
        self._verify_factorisation(_M_TRIDIAGONAL_10, result)

    def test_10x10_arithmetic_rows(self):
        result = smith_normal_form_with_transforms(
            _dense_input(_M_ARITHMETIC_10), backend="magma"
        )
        assert result.invariant_factors == [1, 10]
        self._verify_factorisation(_M_ARITHMETIC_10, result)


# ---------------------------------------------------------------------------
# MAGMA output parsing (line wrapping, etc.)
# ---------------------------------------------------------------------------

class TestMagmaOutputParsing:
    """Test that MAGMA output parser handles line wrapping and continuation."""

    def test_parse_output_with_line_wrapping(self):
        """Verify parser handles very long lines that wrap (e.g., from large integers)."""
        from snforacle.backends.magma import _parse_magma_output

        # Simulate output where a line is so long it wraps due to terminal/buffer limits.
        # When wrapped, the continuation is just a plain line of numbers (no marker).
        # For a 2x2 SNF, LEFT (2x2), RIGHT (2x2):
        # SNF: 4 values, LEFT: 4 values, RIGHT: 4 values
        magma_output = """\
SNF 1 0
 0 2
LEFT 1 0
 0 1
RIGHT 1 0
 0 1
"""
        result = _parse_magma_output(magma_output, nrows=2, ncols=2)

        # Verify the parser correctly collected all wrapped lines
        assert len(result["snf"]) == 2
        assert len(result["left"]) == 2
        assert len(result["right"]) == 2
        assert result["snf"] == [[1, 0], [0, 2]]
        assert result["left"] == [[1, 0], [0, 1]]
        assert result["right"] == [[1, 0], [0, 1]]

    def test_parse_output_with_very_large_integers(self):
        """Verify parser handles very large integers (80+ digit numbers)."""
        from snforacle.backends.magma import _parse_magma_output

        big_num = 658231412009199864494460193206520872909651842522730099580672149990241327855098
        # For a 2x2 matrix: SNF is 2x2 (4 values), LEFT is 2x2 (4 values), RIGHT is 2x2 (4 values)
        magma_output = f"""\
SNF {big_num} 0
 0 {big_num}
LEFT 1 0
 0 1
RIGHT 1 0
 0 1
"""
        result = _parse_magma_output(magma_output, nrows=2, ncols=2)

        # Verify the very large integers were parsed correctly
        assert result["snf"][0][0] == big_num
        assert result["snf"][1][1] == big_num
        assert result["snf"][0][1] == 0
        assert result["snf"][1][0] == 0


# ---------------------------------------------------------------------------
# Cross-backend consistency: magma vs cypari2
# ---------------------------------------------------------------------------

class TestMagmaVsCypari2:
    """Verify that the magma and cypari2 backends produce identical SNF output."""

    def _assert_same(self, entries):
        inp = _dense_input(entries)
        magma_result = smith_normal_form(inp, backend="magma")
        pari_result = smith_normal_form(inp, backend="cypari2")
        assert magma_result.invariant_factors == pari_result.invariant_factors, (
            f"invariant_factors differ: magma={magma_result.invariant_factors} "
            f"cypari2={pari_result.invariant_factors}"
        )
        assert magma_result.smith_normal_form.entries == pari_result.smith_normal_form.entries, (
            f"SNF entries differ: magma={magma_result.smith_normal_form.entries} "
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
