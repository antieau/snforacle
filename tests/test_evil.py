"""Adversarial test suite for snforacle.

This test file is designed to find crashes, incorrect results, unhandled
exceptions, hangs, and surprising behavior in the snforacle package.

Each test class focuses on a specific category of evil input.
"""

import math
import pytest
import sys

from pydantic import ValidationError

from snforacle import (
    DenseIntMatrix,
    SparseIntMatrix,
    smith_normal_form,
    smith_normal_form_with_transforms,
)
from snforacle.schema import SNFResult, SNFWithTransformsResult, SparseEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense(entries, nrows=None, ncols=None):
    """Build a dense dict input. Auto-detects dimensions if not given."""
    if nrows is None:
        nrows = len(entries)
    if ncols is None:
        ncols = len(entries[0]) if entries else 0
    return {"format": "dense", "nrows": nrows, "ncols": ncols, "entries": entries}


def _sparse(nrows, ncols, triples=None):
    """Build a sparse dict input from (row, col, value) triples."""
    if triples is None:
        triples = []
    return {
        "format": "sparse",
        "nrows": nrows,
        "ncols": ncols,
        "entries": [{"row": r, "col": c, "value": v} for r, c, v in triples],
    }


def _mat_mul(A, B):
    """Plain Python matrix multiplication."""
    m, n, p = len(A), len(A[0]), len(B[0])
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
        for i in range(m)
    ]


def _verify_snf_properties(result, nrows, ncols):
    """Verify all SNF structural properties."""
    snf = result.smith_normal_form.entries
    inv = result.invariant_factors

    # Shape check
    assert len(snf) == nrows, f"SNF has {len(snf)} rows, expected {nrows}"
    for i, row in enumerate(snf):
        assert len(row) == ncols, f"SNF row {i} has {len(row)} cols, expected {ncols}"

    # Off-diagonal must be zero
    for i in range(nrows):
        for j in range(ncols):
            if i != j:
                assert snf[i][j] == 0, f"SNF[{i}][{j}] = {snf[i][j]} != 0"

    # Diagonal entries must be non-negative
    diag = [snf[i][i] for i in range(min(nrows, ncols))]
    for i, d in enumerate(diag):
        assert d >= 0, f"SNF diagonal [{i}] = {d} is negative"

    # Divisibility: d_i | d_{i+1}
    nonzero_diag = [d for d in diag if d != 0]
    for i in range(len(nonzero_diag) - 1):
        d_i = nonzero_diag[i]
        d_next = nonzero_diag[i + 1]
        assert d_next % d_i == 0, (
            f"Divisibility violated: d[{i}]={d_i} does not divide d[{i+1}]={d_next}"
        )

    # Invariant factors should match diagonal
    assert inv == nonzero_diag, (
        f"invariant_factors {inv} != nonzero diagonal {nonzero_diag}"
    )

    # Non-decreasing order
    for i in range(len(inv) - 1):
        assert inv[i] <= inv[i + 1], (
            f"Invariant factors not non-decreasing: {inv[i]} > {inv[i+1]}"
        )


def _verify_transforms(M_entries, result, nrows, ncols):
    """Verify U @ M @ V == SNF."""
    U = result.left_transform.entries
    V = result.right_transform.entries
    D = result.smith_normal_form.entries

    # Dimension checks
    assert len(U) == nrows and len(U[0]) == nrows, f"U shape: {len(U)}x{len(U[0])}"
    assert len(V) == ncols and len(V[0]) == ncols, f"V shape: {len(V)}x{len(V[0])}"

    # U @ M @ V == D
    computed = _mat_mul(_mat_mul(U, M_entries), V)
    assert computed == D, f"U@M@V != D\nU@M@V = {computed}\nD = {D}"


# ===========================================================================
# Category 1: Degenerate matrix dimensions
# ===========================================================================

class TestDegenerateDimensions:
    """Test 0x0, 0xN, Nx0, 1x1 matrices."""

    def test_0x0_dense(self):
        """0x0 matrix — no rows, no columns."""
        result = smith_normal_form(_dense([], nrows=0, ncols=0))
        assert result.invariant_factors == []
        assert result.smith_normal_form.entries == []

    def test_0x3_dense(self):
        """0 rows, 3 columns — empty matrix."""
        result = smith_normal_form(_dense([], nrows=0, ncols=3))
        assert result.invariant_factors == []

    def test_3x0_dense(self):
        """3 rows, 0 columns — each row is empty."""
        result = smith_normal_form(_dense([[], [], []], nrows=3, ncols=0))
        assert result.invariant_factors == []

    def test_0x0_with_transforms(self):
        """0x0 with transforms — identity transforms should be 0x0."""
        result = smith_normal_form_with_transforms(_dense([], nrows=0, ncols=0))
        assert result.invariant_factors == []
        assert result.left_transform.entries == []
        assert result.right_transform.entries == []

    def test_0x3_with_transforms(self):
        """0x3 with transforms — left transform 0x0, right 3x3 identity."""
        result = smith_normal_form_with_transforms(_dense([], nrows=0, ncols=3))
        assert result.invariant_factors == []
        assert result.left_transform.nrows == 0
        assert result.right_transform.nrows == 3
        assert result.right_transform.ncols == 3

    def test_3x0_with_transforms(self):
        """3x0 with transforms — left transform 3x3 identity, right 0x0."""
        result = smith_normal_form_with_transforms(
            _dense([[], [], []], nrows=3, ncols=0)
        )
        assert result.invariant_factors == []
        assert result.left_transform.nrows == 3
        assert result.right_transform.nrows == 0

    def test_1x1_zero(self):
        """1x1 zero matrix."""
        result = smith_normal_form(_dense([[0]]))
        assert result.invariant_factors == []
        assert result.smith_normal_form.entries == [[0]]

    def test_1x1_one(self):
        """1x1 identity."""
        result = smith_normal_form(_dense([[1]]))
        assert result.invariant_factors == [1]

    def test_1x1_negative(self):
        """1x1 with negative — SNF should be positive (absolute value)."""
        result = smith_normal_form(_dense([[-7]]))
        assert result.invariant_factors == [7], (
            f"Expected [7] for [[-7]], got {result.invariant_factors}"
        )
        assert result.smith_normal_form.entries == [[7]], (
            f"Expected [[7]], got {result.smith_normal_form.entries}"
        )

    def test_1x1_negative_with_transforms(self):
        """1x1 negative with transforms."""
        result = smith_normal_form_with_transforms(_dense([[-7]]))
        assert result.invariant_factors == [7]
        _verify_transforms([[-7]], result, 1, 1)

    def test_1x1_huge(self):
        """1x1 with a very large integer."""
        big = 2**1000
        result = smith_normal_form(_dense([[big]]))
        assert result.invariant_factors == [big]

    def test_1xN_row_vector(self):
        """1x5 row vector."""
        result = smith_normal_form(_dense([[6, 10, 15]]))
        _verify_snf_properties(result, 1, 3)
        assert result.invariant_factors == [1]

    def test_Nx1_column_vector(self):
        """5x1 column vector."""
        result = smith_normal_form(_dense([[6], [10], [15]]))
        _verify_snf_properties(result, 3, 1)
        assert result.invariant_factors == [1]

    def test_1x1_sparse_zero(self):
        """1x1 sparse zero matrix (empty entries list)."""
        result = smith_normal_form(_sparse(1, 1, []))
        assert result.invariant_factors == []

    def test_0x0_sparse(self):
        """0x0 sparse matrix."""
        result = smith_normal_form(_sparse(0, 0, []))
        assert result.invariant_factors == []


# ===========================================================================
# Category 2: All-zero matrices
# ===========================================================================

class TestAllZeroMatrices:
    """All-zero matrices of various shapes."""

    @pytest.mark.parametrize("nrows,ncols", [(1,1), (2,2), (3,3), (5,5), (2,3), (3,2), (1,5), (5,1)])
    def test_all_zero(self, nrows, ncols):
        entries = [[0]*ncols for _ in range(nrows)]
        result = smith_normal_form(_dense(entries))
        assert result.invariant_factors == []
        _verify_snf_properties(result, nrows, ncols)

    @pytest.mark.parametrize("backend", ["cypari2", "flint"])
    def test_all_zero_backends(self, backend):
        """Cross-backend consistency for zero matrices."""
        entries = [[0]*3 for _ in range(3)]
        result = smith_normal_form(_dense(entries), backend=backend)
        assert result.invariant_factors == []

    def test_all_zero_with_transforms(self):
        """3x3 zero matrix with transforms."""
        M = [[0,0,0],[0,0,0],[0,0,0]]
        result = smith_normal_form_with_transforms(_dense(M))
        assert result.invariant_factors == []
        _verify_transforms(M, result, 3, 3)


# ===========================================================================
# Category 3: Identity matrices
# ===========================================================================

class TestIdentityMatrices:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    def test_identity(self, n):
        I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        result = smith_normal_form(_dense(I))
        assert result.invariant_factors == [1]*n

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_identity_with_transforms(self, n):
        I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        result = smith_normal_form_with_transforms(_dense(I))
        assert result.invariant_factors == [1]*n
        _verify_transforms(I, result, n, n)


# ===========================================================================
# Category 4: Extreme integer values
# ===========================================================================

class TestExtremeValues:
    """Very large, very negative, and mixed-sign integers."""

    def test_large_positive(self):
        """Matrix with entries of magnitude ~2^100."""
        big = 2**100
        M = [[big, 0], [0, big]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [big, big]

    def test_large_negative(self):
        """Matrix with large negative entries."""
        big = -(2**100)
        M = [[big, 0], [0, big]]
        result = smith_normal_form(_dense(M))
        # SNF entries should be positive
        _verify_snf_properties(result, 2, 2)

    def test_very_large_2x2(self):
        """2x2 with 1000-bit integers."""
        a = 2**1000
        b = 3**630  # roughly comparable in magnitude
        M = [[a, b], [b, a]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 2, 2)

    def test_mixed_huge_and_tiny(self):
        """Matrix mixing huge and small integers."""
        M = [[2**500, 1], [0, 2**500]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 2, 2)
        assert result.invariant_factors == [1, 2**1000]

    def test_large_determinant_transforms(self):
        """Verify transforms with large-integer matrix."""
        M = [[2**100, 0], [0, 3**63]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_transforms(M, result, 2, 2)

    @pytest.mark.parametrize("backend", ["cypari2", "flint"])
    def test_large_cross_backend(self, backend):
        """Large integer cross-backend consistency."""
        M = [[2**100, 0], [0, 3**63]]
        result = smith_normal_form(_dense(M), backend=backend)
        _verify_snf_properties(result, 2, 2)

    def test_negative_diagonal(self):
        """Diagonal matrix with negative entries — SNF should have positive diagonal."""
        M = [[-6, 0, 0], [0, -15, 0], [0, 0, -10]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)
        # gcd(6,15,10)=1, lcm structure: 1, 30, 30... let's just check positivity + divisibility
        for d in result.invariant_factors:
            assert d > 0, f"SNF factor should be positive, got {d}"


# ===========================================================================
# Category 5: Non-square matrices
# ===========================================================================

class TestNonSquareMatrices:
    """Tall, wide, 1xN, Nx1 matrices."""

    def test_2x5_wide(self):
        M = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 2, 5)

    def test_5x2_tall(self):
        M = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 5, 2)

    def test_2x5_with_transforms(self):
        M = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf_properties(result, 2, 5)
        _verify_transforms(M, result, 2, 5)

    def test_5x2_with_transforms(self):
        M = [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf_properties(result, 5, 2)
        _verify_transforms(M, result, 5, 2)

    def test_1x10(self):
        M = [[2, 4, 6, 8, 10, 12, 14, 16, 18, 20]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 1, 10)
        assert result.invariant_factors == [2]

    def test_10x1(self):
        M = [[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 10, 1)
        assert result.invariant_factors == [2]

    def test_1x1_with_transforms(self):
        M = [[42]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_transforms(M, result, 1, 1)
        assert result.invariant_factors == [42]


# ===========================================================================
# Category 6: Sparse matrix edge cases
# ===========================================================================

class TestSparseEdgeCases:
    """Edge cases specific to sparse input format."""

    def test_empty_entries_3x3(self):
        """Sparse 3x3 with no entries = zero matrix."""
        result = smith_normal_form(_sparse(3, 3, []))
        assert result.invariant_factors == []

    def test_single_entry(self):
        """Sparse 3x3 with exactly one nonzero entry."""
        result = smith_normal_form(_sparse(3, 3, [(1, 2, 7)]))
        _verify_snf_properties(result, 3, 3)
        assert result.invariant_factors == [7]

    def test_sparse_entry_value_zero(self):
        """Sparse entry with value=0 — should be accepted but treated as zero."""
        result = smith_normal_form(_sparse(2, 2, [(0, 0, 0)]))
        assert result.invariant_factors == []

    def test_negative_row_index(self):
        """Negative row index in sparse entry — should be rejected."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_sparse(3, 3, [(-1, 0, 5)]))

    def test_negative_col_index(self):
        """Negative col index in sparse entry — should be rejected."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_sparse(3, 3, [(0, -1, 5)]))

    def test_row_out_of_bounds(self):
        """Row index >= nrows."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_sparse(3, 3, [(3, 0, 5)]))

    def test_col_out_of_bounds(self):
        """Col index >= ncols."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_sparse(3, 3, [(0, 3, 5)]))

    def test_duplicate_entries(self):
        """Duplicate (row, col) in sparse entries."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_sparse(3, 3, [(0, 0, 1), (0, 0, 2)]))

    def test_sparse_0x0(self):
        """0x0 sparse matrix."""
        result = smith_normal_form(_sparse(0, 0, []))
        assert result.invariant_factors == []

    def test_sparse_0x3(self):
        """0x3 sparse matrix."""
        result = smith_normal_form(_sparse(0, 3, []))
        assert result.invariant_factors == []

    def test_sparse_3x0(self):
        """3x0 sparse matrix."""
        result = smith_normal_form(_sparse(3, 0, []))
        assert result.invariant_factors == []

    def test_sparse_large_value(self):
        """Sparse entry with huge integer value."""
        big = 2**500
        result = smith_normal_form(_sparse(2, 2, [(0, 0, big), (1, 1, big)]))
        assert result.invariant_factors == [big, big]

    def test_sparse_negative_value(self):
        """Sparse entry with negative value."""
        result = smith_normal_form(_sparse(1, 1, [(0, 0, -42)]))
        _verify_snf_properties(result, 1, 1)

    def test_sparse_with_transforms(self):
        """Sparse input with transforms."""
        entries = [(0, 0, 2), (0, 1, 4), (1, 0, 6), (1, 1, 8)]
        result = smith_normal_form_with_transforms(_sparse(2, 2, entries))
        _verify_snf_properties(result, 2, 2)
        # Reconstruct dense for transform verification
        M = [[0]*2 for _ in range(2)]
        for r, c, v in entries:
            M[r][c] = v
        _verify_transforms(M, result, 2, 2)


# ===========================================================================
# Category 7: Invalid inputs — type errors, missing fields, malformed data
# ===========================================================================

class TestInvalidInputs:
    """Inputs that should be cleanly rejected with helpful error messages."""

    def test_missing_format(self):
        """Dict without 'format' key."""
        with pytest.raises((ValidationError, ValueError, KeyError)):
            smith_normal_form({"nrows": 2, "ncols": 2, "entries": [[1,0],[0,1]]})

    def test_wrong_format_string(self):
        """Format is neither 'dense' nor 'sparse'."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form({
                "format": "csc",
                "nrows": 2, "ncols": 2,
                "entries": [[1,0],[0,1]],
            })

    def test_float_entries(self):
        """Float values in entries — should be rejected or coerced."""
        # Pydantic int field may coerce 1.0 -> 1 but reject 1.5
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_dense([[1.5, 0], [0, 1]]))

    def test_float_coercion(self):
        """Float 1.0 might be silently coerced to int 1 — test behavior."""
        try:
            result = smith_normal_form(_dense([[1.0, 0.0], [0.0, 1.0]]))
            # If it succeeds, check correctness
            assert result.invariant_factors == [1, 1]
        except (ValidationError, ValueError):
            pass  # Also acceptable: strict rejection

    def test_string_entries(self):
        """String values in entries."""
        with pytest.raises((ValidationError, ValueError, TypeError)):
            smith_normal_form(_dense([["a", "b"], ["c", "d"]]))

    def test_none_entries(self):
        """None in entries."""
        with pytest.raises((ValidationError, ValueError, TypeError)):
            smith_normal_form(_dense([[None, 0], [0, 1]]))

    def test_none_matrix(self):
        """None as the entire matrix argument."""
        with pytest.raises((ValidationError, ValueError, TypeError, AttributeError)):
            smith_normal_form(None)

    def test_empty_list(self):
        """Bare empty list (no format key)."""
        with pytest.raises((ValidationError, ValueError, TypeError, AttributeError)):
            smith_normal_form([])

    def test_nested_too_deep(self):
        """Entries nested one level too deep."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_dense([[[1, 2]], [[3, 4]]]))

    def test_non_rectangular_entries(self):
        """Rows of different lengths."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form({
                "format": "dense", "nrows": 2, "ncols": 2,
                "entries": [[1, 2], [3]],
            })

    def test_negative_nrows(self):
        """Negative nrows."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form({
                "format": "dense", "nrows": -1, "ncols": 2,
                "entries": [],
            })

    def test_negative_ncols(self):
        """Negative ncols."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form({
                "format": "dense", "nrows": 2, "ncols": -1,
                "entries": [[1, 2], [3, 4]],
            })

    def test_nrows_mismatch(self):
        """nrows doesn't match len(entries)."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form({
                "format": "dense", "nrows": 3, "ncols": 2,
                "entries": [[1, 2], [3, 4]],
            })

    def test_ncols_mismatch(self):
        """ncols doesn't match actual row lengths."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form({
                "format": "dense", "nrows": 2, "ncols": 3,
                "entries": [[1, 2], [3, 4]],
            })


# ===========================================================================
# Category 8: Backend-specific edge cases
# ===========================================================================

class TestBackendEdgeCases:
    """Backend-specific issues."""

    def test_unknown_backend(self):
        """Non-existent backend name."""
        with pytest.raises(ValueError, match="Unknown backend"):
            smith_normal_form(_dense([[1, 0], [0, 1]]), backend="nonexistent")

    def test_flint_no_transforms(self):
        """FLINT backend doesn't support transforms."""
        with pytest.raises(NotImplementedError):
            smith_normal_form_with_transforms(
                _dense([[1, 0], [0, 1]]), backend="flint"
            )

    def test_empty_string_backend(self):
        """Empty string as backend name."""
        with pytest.raises(ValueError):
            smith_normal_form(_dense([[1]]), backend="")

    @pytest.mark.parametrize("backend", ["cypari2", "flint"])
    def test_1x1_zero_backend(self, backend):
        """1x1 zero matrix across backends."""
        result = smith_normal_form(_dense([[0]]), backend=backend)
        assert result.invariant_factors == []

    @pytest.mark.parametrize("backend", ["cypari2", "flint"])
    def test_1x1_negative_backend(self, backend):
        """1x1 negative across backends — should give positive SNF."""
        result = smith_normal_form(_dense([[-5]]), backend=backend)
        assert result.invariant_factors == [5], (
            f"Backend {backend}: expected [5], got {result.invariant_factors}"
        )
        assert result.smith_normal_form.entries == [[5]], (
            f"Backend {backend}: expected [[5]], got {result.smith_normal_form.entries}"
        )


# ===========================================================================
# Category 9: Mathematical edge cases
# ===========================================================================

class TestMathematicalEdgeCases:
    """Cases designed to test mathematical correctness."""

    def test_gcd_complications(self):
        """Matrix where GCD structure is non-trivial."""
        # [[6, 0], [0, 10]] — SNF should be diag(2, 30)
        M = [[6, 0], [0, 10]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 2, 2)
        assert result.invariant_factors == [2, 30]

    def test_divisibility_chain(self):
        """Matrix whose SNF has a non-trivial divisibility chain."""
        # diag(2, 6, 30) — already in SNF form: 2|6|30
        M = [[2, 0, 0], [0, 6, 0], [0, 0, 30]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [2, 6, 30]

    def test_non_trivial_snf(self):
        """Matrix where the SNF is not obvious from the entries."""
        # [[2, 3], [4, 6]] — rank 1, gcd(2,3,4,6) = 1, so SNF = diag(1, 0)
        M = [[2, 3], [4, 6]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 2, 2)
        assert result.invariant_factors == [1]

    def test_repeated_invariant_factors(self):
        """Matrix with repeated invariant factors."""
        # diag(2, 2, 2) — SNF should be diag(2, 2, 2)
        M = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [2, 2, 2]

    def test_prime_power_structure(self):
        """Matrix with prime power structure in SNF."""
        # diag(2, 4, 8) — already divisibility chain
        M = [[2, 0, 0], [0, 4, 0], [0, 0, 8]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [2, 4, 8]

    def test_permutation_matrix(self):
        """Permutation matrix — SNF should be identity."""
        # cycle (123): [[0,1,0],[0,0,1],[1,0,0]]
        M = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1, 1]

    def test_nilpotent(self):
        """Nilpotent matrix — SNF should be all zeros."""
        # [[0,1,0],[0,0,1],[0,0,0]]
        M = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)
        # This matrix has rank 2, SNF should be diag(1, 1, 0)
        assert result.invariant_factors == [1, 1]

    def test_companion_matrix(self):
        """Companion matrix of x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)."""
        M = [[0, 0, 6], [1, 0, -11], [0, 1, 6]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)

    def test_single_row_gcd(self):
        """1x3 matrix [[6, 10, 15]] — SNF should be [[1, 0, 0]] since gcd(6,10,15) = 1."""
        M = [[6, 10, 15]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1]

    def test_single_col_gcd(self):
        """3x1 matrix [[6],[10],[15]] — SNF should be [[1],[0],[0]]."""
        M = [[6], [10], [15]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1]

    def test_coprime_entries(self):
        """Matrix with coprime entries."""
        # [[2, 3], [5, 7]] — det = -1, so SNF = I
        M = [[2, 3], [5, 7]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1]

    def test_diagonal_not_in_snf_order(self):
        """Diagonal matrix not in SNF order: diag(6, 2, 4)."""
        M = [[6, 0, 0], [0, 2, 0], [0, 0, 4]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)
        assert result.invariant_factors == [2, 2, 12]

    def test_unimodular_matrix(self):
        """Matrix with det = +/-1 — SNF is identity."""
        M = [[1, 2], [0, 1]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1]

    def test_non_square_rank_deficient(self):
        """Non-square rank-deficient matrix."""
        # [[1, 2, 3], [2, 4, 6]] — rank 1, gcd = 1
        M = [[1, 2, 3], [2, 4, 6]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 2, 3)
        assert result.invariant_factors == [1]


# ===========================================================================
# Category 10: Cross-backend consistency
# ===========================================================================

class TestCrossBackendConsistency:
    """Same input should give same SNF across all available backends."""

    MATRICES = [
        ("identity_2x2", [[1, 0], [0, 1]]),
        ("zero_2x2", [[0, 0], [0, 0]]),
        ("standard_3x3", [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]),
        ("rank_deficient", [[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ("non_square_2x3", [[1, 2, 3], [4, 5, 6]]),
        ("non_square_3x2", [[1, 4], [2, 5], [3, 6]]),
        ("negative", [[-6, 0], [0, -10]]),
        ("1x1_positive", [[42]]),
        ("1x1_negative", [[-42]]),
        ("1x1_zero", [[0]]),
        ("single_row", [[6, 10, 15]]),
        ("single_col", [[6], [10], [15]]),
        ("diagonal_unordered", [[6, 0, 0], [0, 2, 0], [0, 0, 4]]),
    ]

    @pytest.mark.parametrize("name,M", MATRICES, ids=[m[0] for m in MATRICES])
    def test_cypari2_vs_flint(self, name, M):
        """cypari2 and flint should give identical results."""
        r1 = smith_normal_form(_dense(M), backend="cypari2")
        r2 = smith_normal_form(_dense(M), backend="flint")
        assert r1.invariant_factors == r2.invariant_factors, (
            f"Matrix '{name}': cypari2={r1.invariant_factors} vs flint={r2.invariant_factors}"
        )
        assert r1.smith_normal_form.entries == r2.smith_normal_form.entries, (
            f"Matrix '{name}': SNF entries differ"
        )


# ===========================================================================
# Category 11: Transform correctness (extensive)
# ===========================================================================

class TestTransformCorrectnessExtensive:
    """Extensive tests that U @ M @ V == SNF for many matrix types."""

    TRANSFORM_MATRICES = [
        ("identity_3x3", [[1,0,0],[0,1,0],[0,0,1]], 3, 3),
        ("zero_3x3", [[0,0,0],[0,0,0],[0,0,0]], 3, 3),
        ("standard_3x3", [[2,4,4],[-6,6,12],[10,-4,-16]], 3, 3),
        ("rank_1", [[1,2,3],[2,4,6],[3,6,9]], 3, 3),
        ("non_square_2x3", [[1,2,3],[4,5,6]], 2, 3),
        ("non_square_3x2", [[1,4],[2,5],[3,6]], 3, 2),
        ("non_square_4x2", [[1,0],[0,1],[1,1],[2,3]], 4, 2),
        ("non_square_2x4", [[1,0,1,2],[0,1,1,3]], 2, 4),
        ("1x1_pos", [[42]], 1, 1),
        ("1x1_neg", [[-42]], 1, 1),
        ("1x1_zero", [[0]], 1, 1),
        ("1x3_row", [[6, 10, 15]], 1, 3),
        ("3x1_col", [[6],[10],[15]], 3, 1),
        ("negative_diagonal", [[-6,0,0],[0,-15,0],[0,0,-10]], 3, 3),
        ("unimodular", [[1,2],[0,1]], 2, 2),
        ("permutation", [[0,1,0],[0,0,1],[1,0,0]], 3, 3),
        ("large_spread", [[1,0],[0,2**50]], 2, 2),
    ]

    @pytest.mark.parametrize("name,M,nrows,ncols", TRANSFORM_MATRICES,
                             ids=[t[0] for t in TRANSFORM_MATRICES])
    def test_transform_correctness(self, name, M, nrows, ncols):
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf_properties(result, nrows, ncols)
        _verify_transforms(M, result, nrows, ncols)


# ===========================================================================
# Category 12: Negative invariant factors — the BIG one
# ===========================================================================

class TestNegativeInvariantFactors:
    """PARI/GP may return negative invariant factors. The backend must ensure
    all SNF diagonal entries are non-negative."""

    NEGATIVE_MATRICES = [
        [[-1]],
        [[-1, 0], [0, -1]],
        [[-2, -3], [-5, -7]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-6, 0], [0, 10]],
        [[0, -1], [1, 0]],  # rotation matrix, det = 1
    ]

    @pytest.mark.parametrize("M", NEGATIVE_MATRICES)
    def test_snf_entries_non_negative_cypari2(self, M):
        """cypari2 backend: all SNF diagonal entries must be >= 0."""
        result = smith_normal_form(_dense(M), backend="cypari2")
        for d in result.invariant_factors:
            assert d >= 0, f"Negative invariant factor {d} from cypari2 for matrix {M}"
        snf = result.smith_normal_form.entries
        nrows, ncols = len(M), len(M[0])
        for i in range(min(nrows, ncols)):
            assert snf[i][i] >= 0, f"Negative SNF diagonal [{i}] = {snf[i][i]}"

    @pytest.mark.parametrize("M", NEGATIVE_MATRICES)
    def test_snf_entries_non_negative_flint(self, M):
        """flint backend: all SNF diagonal entries must be >= 0."""
        result = smith_normal_form(_dense(M), backend="flint")
        for d in result.invariant_factors:
            assert d >= 0, f"Negative invariant factor {d} from flint for matrix {M}"


# ===========================================================================
# Category 13: Diagonal with zeros intermixed
# ===========================================================================

class TestDiagonalWithZeros:
    """Matrices that are already diagonal but have zeros in non-bottom positions."""

    def test_diag_1_0_2(self):
        """diag(1, 0, 2) — SNF should be diag(1, 2, 0), not diag(1, 0, 2)."""
        M = [[1, 0, 0], [0, 0, 0], [0, 0, 2]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)
        assert result.invariant_factors == [1, 2]
        snf = result.smith_normal_form.entries
        # Zeros should be at the end
        assert snf[0][0] == 1
        assert snf[1][1] == 2
        assert snf[2][2] == 0

    def test_diag_0_1_0(self):
        """diag(0, 1, 0) — SNF should be diag(1, 0, 0)."""
        M = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)
        assert result.invariant_factors == [1]

    def test_diag_6_0_4(self):
        """diag(6, 0, 4) — SNF should be diag(2, 12, 0)."""
        M = [[6, 0, 0], [0, 0, 0], [0, 0, 4]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)
        assert result.invariant_factors == [2, 12]


# ===========================================================================
# Category 14: Matrices with repeated diagonal entries
# ===========================================================================

class TestRepeatedDiagonalEntries:
    """Test _permutation_matrices when D_pari has duplicate values."""

    def test_diag_2_2_2_transforms(self):
        """All diagonal entries identical — permutation matching must handle ties."""
        M = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_transforms(M, result, 3, 3)

    def test_diag_3_3_transforms(self):
        """2x2 with repeated factors."""
        M = [[3, 0], [0, 3]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_transforms(M, result, 2, 2)

    def test_mixed_repeated_transforms(self):
        """Matrix with SNF having repeated factors: diag(1, 2, 2, 4)."""
        # Construct a matrix with known SNF = diag(1, 2, 2, 4)
        # Use diag(1, 2, 2, 4) directly
        M = [[1,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,4]]
        result = smith_normal_form_with_transforms(_dense(M))
        assert result.invariant_factors == [1, 2, 2, 4]
        _verify_transforms(M, result, 4, 4)


# ===========================================================================
# Category 15: Sparse entry with value exactly 0
# ===========================================================================

class TestSparseZeroValueEntries:
    """Sparse entries with value=0 — should these be accepted?"""

    def test_zero_value_entry(self):
        """Entry (0,0, value=0) — the matrix is all-zero."""
        result = smith_normal_form(_sparse(2, 2, [(0, 0, 0)]))
        assert result.invariant_factors == []

    def test_zero_and_nonzero_mixed(self):
        """Mix of zero and nonzero sparse entries."""
        result = smith_normal_form(_sparse(2, 2, [(0, 0, 0), (1, 1, 5)]))
        assert result.invariant_factors == [5]


# ===========================================================================
# Category 16: Pydantic model direct construction
# ===========================================================================

class TestPydanticDirectConstruction:
    """Test that Pydantic models can be passed directly (not just dicts)."""

    def test_dense_model_input(self):
        model = DenseIntMatrix(format="dense", nrows=2, ncols=2, entries=[[1,0],[0,1]])
        result = smith_normal_form(model)
        assert result.invariant_factors == [1, 1]

    def test_sparse_model_input(self):
        model = SparseIntMatrix(
            format="sparse", nrows=2, ncols=2,
            entries=[SparseEntry(row=0, col=0, value=5)]
        )
        result = smith_normal_form(model)
        assert result.invariant_factors == [5]


# ===========================================================================
# Category 17: _extract_invariant_factors sign handling
# ===========================================================================

class TestInvariantFactorSigns:
    """PARI's matsnf can return negative factors. Test that the code handles this."""

    def test_negative_1x1_cypari2(self):
        """[[-1]] through cypari2 — should give invariant factor [1], not [-1]."""
        result = smith_normal_form(_dense([[-1]]), backend="cypari2")
        assert result.invariant_factors == [1]
        assert result.smith_normal_form.entries == [[1]]

    def test_anti_diagonal(self):
        """Anti-diagonal matrix [[0, 1], [1, 0]] — det = -1, SNF = I."""
        M = [[0, 1], [1, 0]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1]

    def test_negative_determinant_3x3(self):
        """Matrix with negative determinant."""
        # det([[1,0,0],[0,1,0],[0,0,-1]]) = -1
        M = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1, 1]
        _verify_snf_properties(result, 3, 3)

    def test_all_negative_entries(self):
        """Matrix with all negative entries."""
        M = [[-2, -4, -4], [-6, -6, -12], [-10, -4, -16]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)
        for d in result.invariant_factors:
            assert d > 0


# ===========================================================================
# Category 18: matsnf vector length edge case
# ===========================================================================

class TestMatsnfVectorLength:
    """PARI's matsnf(flag=0) returns a vector of length min(nrows, ncols)
    for square matrices, but behavior may differ for non-square.
    Test that _extract_invariant_factors handles this correctly."""

    def test_wide_matrix_invariant_factors(self):
        """2x5 matrix — only up to 2 invariant factors possible."""
        M = [[1, 0, 0, 0, 0], [0, 6, 0, 0, 0]]
        result = smith_normal_form(_dense(M), backend="cypari2")
        _verify_snf_properties(result, 2, 5)
        assert result.invariant_factors == [1, 6]

    def test_tall_matrix_invariant_factors(self):
        """5x2 matrix — only up to 2 invariant factors possible."""
        M = [[1, 0], [0, 6], [0, 0], [0, 0], [0, 0]]
        result = smith_normal_form(_dense(M), backend="cypari2")
        _verify_snf_properties(result, 5, 2)
        assert result.invariant_factors == [1, 6]


# ===========================================================================
# Category 19: Computation blowup / performance
# ===========================================================================

class TestPerformance:
    """Matrices that might cause intermediate computation blowup."""

    def test_hilbert_like_3x3(self):
        """Matrix with entries 1/(i+j+1) scaled up — can cause large intermediates."""
        # H_3 scaled by lcm(1..6) = 60
        M = [[60, 30, 20], [30, 20, 15], [20, 15, 12]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 3, 3)

    @pytest.mark.parametrize("backend", ["cypari2", "flint"])
    def test_20x20_identity(self, backend):
        """20x20 identity — should be fast."""
        n = 20
        I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        result = smith_normal_form(_dense(I), backend=backend)
        assert result.invariant_factors == [1]*n

    @pytest.mark.parametrize("backend", ["cypari2", "flint"])
    def test_20x20_random_like(self, backend):
        """20x20 matrix with structured entries — should complete."""
        n = 20
        M = [[(i*n + j + 1) % 97 for j in range(n)] for i in range(n)]
        result = smith_normal_form(_dense(M), backend=backend)
        _verify_snf_properties(result, n, n)


# ===========================================================================
# Category 20: Invariant factor absolute value consistency
# ===========================================================================

class TestAbsoluteValueConsistency:
    """Ensure PARI's sign convention doesn't leak through."""

    def test_matsnf_raw_sign(self):
        """Directly test that _extract_invariant_factors returns positive values."""
        from snforacle.backends.cypari2 import _pari, _extract_invariant_factors

        pari = _pari()
        # Matrix [[-1]] — PARI's matsnf may return [-1]
        mat = pari.matrix(1, 1, [-1])
        factors = _extract_invariant_factors(mat)
        # This is the key test: are factors positive?
        for f in factors:
            if f < 0:
                pytest.fail(
                    f"_extract_invariant_factors returned negative factor {f} "
                    f"for matrix [[-1]]. Raw matsnf output: {mat.matsnf()}"
                )

    def test_matsnf_raw_sign_3x3(self):
        """Test PARI raw output for a matrix with negative determinant."""
        from snforacle.backends.cypari2 import _pari, _extract_invariant_factors

        pari = _pari()
        mat = pari.matrix(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, -1])
        factors = _extract_invariant_factors(mat)
        for f in factors:
            if f < 0:
                pytest.fail(
                    f"_extract_invariant_factors returned negative factor {f} "
                    f"for diag(1,1,-1). Raw: {mat.matsnf()}"
                )

    def test_matsnf_negative_det_2x2(self):
        """[[0, 1], [-1, 0]] has det=1 but entries are negative."""
        from snforacle.backends.cypari2 import _pari, _extract_invariant_factors

        pari = _pari()
        mat = pari.matrix(2, 2, [0, 1, -1, 0])
        factors = _extract_invariant_factors(mat)
        assert all(f > 0 for f in factors), f"Got negative factors: {factors}"


# ===========================================================================
# Category 21: Edge cases in _permutation_matrices
# ===========================================================================

class TestPermutationMatricesEdgeCases:
    """The _permutation_matrices function matches D_pari to D_std by value.
    Edge cases include: all-zero matrix, repeated values, single nonzero."""

    def test_all_zero_no_permutation_needed(self):
        """All-zero matrix — permutation matrices should be identity (or any valid permutation)."""
        M = [[0, 0], [0, 0]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_transforms(M, result, 2, 2)

    def test_single_nonzero_rectangular(self):
        """2x3 matrix with one nonzero entry."""
        M = [[0, 0, 5], [0, 0, 0]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_transforms(M, result, 2, 3)

    def test_all_same_diagonal_permutation(self):
        """4x4 diagonal with all same values — permutation matching with ties."""
        M = [[5,0,0,0],[0,5,0,0],[0,0,5,0],[0,0,0,5]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_transforms(M, result, 4, 4)


# ===========================================================================
# Category 22: Serialization round-trip
# ===========================================================================

class TestSerializationRoundTrip:
    """Test that results survive JSON serialization."""

    def test_snf_result_json_round_trip(self):
        result = smith_normal_form(_dense([[2, 4], [6, 8]]))
        json_str = result.model_dump_json()
        restored = SNFResult.model_validate_json(json_str)
        assert restored.invariant_factors == result.invariant_factors

    def test_transforms_result_json_round_trip(self):
        result = smith_normal_form_with_transforms(_dense([[2, 4], [6, 8]]))
        json_str = result.model_dump_json()
        restored = SNFWithTransformsResult.model_validate_json(json_str)
        assert restored.invariant_factors == result.invariant_factors
        assert restored.left_transform.entries == result.left_transform.entries
        assert restored.right_transform.entries == result.right_transform.entries

    def test_large_integer_json_round_trip(self):
        """Large integers must survive JSON serialization."""
        big = 2**500
        result = smith_normal_form(_dense([[big, 0], [0, big]]))
        json_str = result.model_dump_json()
        restored = SNFResult.model_validate_json(json_str)
        assert restored.invariant_factors == [big, big]


# ===========================================================================
# Category 23: Stress test — matrices known to exercise PARI quirks
# ===========================================================================

class TestPariQuirks:
    """Matrices that exercise known PARI/GP matsnf edge cases."""

    def test_pari_non_square_ordering(self):
        """PARI may order non-square SNF differently. Verify normalization."""
        # 3x5 matrix with known SNF
        M = [
            [2, 0, 0, 0, 0],
            [0, 6, 0, 0, 0],
            [0, 0, 12, 0, 0],
        ]
        result = smith_normal_form(_dense(M), backend="cypari2")
        _verify_snf_properties(result, 3, 5)
        assert result.invariant_factors == [2, 6, 12]

    def test_pari_non_square_transforms(self):
        """Non-square transforms through PARI — common source of dimension bugs."""
        M = [
            [2, 0, 0, 0, 0],
            [0, 6, 0, 0, 0],
            [0, 0, 12, 0, 0],
        ]
        result = smith_normal_form_with_transforms(_dense(M), backend="cypari2")
        _verify_snf_properties(result, 3, 5)
        _verify_transforms(M, result, 3, 5)

    def test_pari_wide_rank_deficient(self):
        """Wide rank-deficient matrix."""
        M = [[1, 2, 3, 4], [2, 4, 6, 8]]
        result = smith_normal_form_with_transforms(_dense(M), backend="cypari2")
        _verify_snf_properties(result, 2, 4)
        _verify_transforms(M, result, 2, 4)

    def test_pari_tall_rank_deficient(self):
        """Tall rank-deficient matrix."""
        M = [[1, 2], [2, 4], [3, 6], [4, 8]]
        result = smith_normal_form_with_transforms(_dense(M), backend="cypari2")
        _verify_snf_properties(result, 4, 2)
        _verify_transforms(M, result, 4, 2)


# ===========================================================================
# Category 24: _mat_mul edge cases
# ===========================================================================

class TestMatMulEdgeCases:
    """Test the helper _mat_mul function used in cypari2 backend."""

    def test_mat_mul_1x1(self):
        from snforacle.backends.cypari2 import _mat_mul
        assert _mat_mul([[3]], [[7]]) == [[21]]

    def test_mat_mul_non_square(self):
        from snforacle.backends.cypari2 import _mat_mul
        # (2x3) @ (3x2) = (2x2)
        A = [[1, 2, 3], [4, 5, 6]]
        B = [[7, 8], [9, 10], [11, 12]]
        result = _mat_mul(A, B)
        assert result == [[58, 64], [139, 154]]


# ===========================================================================
# Category 25: Tricky divisibility matrices
# ===========================================================================

class TestTrickyDivisibility:
    """Matrices where naive algorithms fail the d_i | d_{i+1} property."""

    def test_classic_snf_example(self):
        """The classic SNF example: diag entries aren't obvious."""
        # [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        # SNF = diag(2, 6, 12), and 2|6 and 6|12
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [2, 6, 12]

    def test_gcd_not_an_entry(self):
        """Matrix where the first invariant factor (the gcd) is not an entry."""
        # [[6, 10], [10, 15]] — gcd of all entries is 1, but no entry is 1
        M = [[6, 10], [10, 15]]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 2, 2)
        # det = 6*15 - 10*10 = 90 - 100 = -10
        # SNF = diag(1, 10)
        assert result.invariant_factors == [1, 10]

    def test_large_invariant_factor_ratio(self):
        """Large ratio between consecutive invariant factors."""
        # diag(1, 10^50) — ratio 10^50
        big = 10**50
        M = [[1, 0], [0, big]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, big]


# ===========================================================================
# Category 26: Large matrices with large SNF values (output parsing stress)
# ===========================================================================

class TestLargeMatricesWithLargeSNF:
    """Test matrices that combine:
    - Moderately-sized dimensions (10x10, 20x20)
    - With large enough entries/SNF values to produce very long output lines

    This catches issues in backend output parsing (e.g., line wrapping in MAGMA).
    """

    def test_10x10_with_large_entries(self):
        """10x10 matrix with 100-bit entries — produces long SNF output."""
        import random
        random.seed(42)
        big = 2**100
        M = [[random.randint(0, big) for _ in range(10)] for _ in range(10)]
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 10, 10)
        # Just verify it doesn't crash; actual values are backend-dependent

    def test_15x15_powers_of_two(self):
        """15x15 diagonal matrix with large powers of 2."""
        M = [[0] * 15 for _ in range(15)]
        for i in range(15):
            M[i][i] = 2 ** (64 + i)  # Very large diagonal entries
        result = smith_normal_form(_dense(M))
        assert len(result.invariant_factors) == 15
        # SNF should be the same as the input (diagonal with these exact values)
        expected = [2 ** (64 + i) for i in range(15)]
        assert result.invariant_factors == expected

    def test_20x20_with_large_snf_values(self):
        """20x20 matrix where SNF computation produces large intermediate values."""
        # Use a Fibonacci-like matrix; determinant grows exponentially
        M = [[0] * 20 for _ in range(20)]
        for i in range(20):
            for j in range(20):
                if i == j:
                    M[i][j] = 2
                elif j == i + 1:
                    M[i][j] = 1
        result = smith_normal_form(_dense(M))
        _verify_snf_properties(result, 20, 20)

    def test_large_matrix_with_transforms(self):
        """Large matrix (15x15) with transforms — tests full output parsing."""
        import random
        random.seed(123)
        big = 2**80
        M = [[random.randint(1, big) for _ in range(15)] for _ in range(15)]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf_properties(result, 15, 15)
        _verify_transforms(M, result, 15, 15)


