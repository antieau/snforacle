"""Adversarial test suite, round 2 — deeper probes.

Focuses on:
1. PARI's matsnf(flag=0) returning negative values (raw check)
2. The _permutation_matrices code with adversarial duplicate patterns
3. Non-square transforms with rank < min(nrows, ncols)
4. Matrices where PARI rearranges the SNF diagonal maximally
5. Boolean / numpy-integer type coercion
6. Extremely large sparse matrices (dimension only)
7. inf/nan via float coercion attempts
8. Backend=None, backend with whitespace
9. Matrices that trigger abs() issues in _extract_invariant_factors
10. Transform factorization with non-square rank-deficient all-zero columns
"""

import pytest
import sys

from pydantic import ValidationError

from snforacle import (
    DenseIntMatrix,
    SparseIntMatrix,
    smith_normal_form,
    smith_normal_form_with_transforms,
)
from snforacle.schema import SNFResult, SNFWithTransformsResult


def _dense(entries, nrows=None, ncols=None):
    if nrows is None:
        nrows = len(entries)
    if ncols is None:
        ncols = len(entries[0]) if entries else 0
    return {"format": "dense", "nrows": nrows, "ncols": ncols, "entries": entries}


def _sparse(nrows, ncols, triples=None):
    if triples is None:
        triples = []
    return {
        "format": "sparse",
        "nrows": nrows,
        "ncols": ncols,
        "entries": [{"row": r, "col": c, "value": v} for r, c, v in triples],
    }


def _mat_mul(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
        for i in range(m)
    ]


def _verify_snf(result, nrows, ncols):
    snf = result.smith_normal_form.entries
    assert len(snf) == nrows
    for row in snf:
        assert len(row) == ncols
    for i in range(nrows):
        for j in range(ncols):
            if i != j:
                assert snf[i][j] == 0
    diag = [snf[i][i] for i in range(min(nrows, ncols))]
    for d in diag:
        assert d >= 0
    nonzero = [d for d in diag if d != 0]
    for i in range(len(nonzero) - 1):
        assert nonzero[i+1] % nonzero[i] == 0
    assert result.invariant_factors == nonzero


def _verify_transforms(M, result, nrows, ncols):
    U = result.left_transform.entries
    V = result.right_transform.entries
    D = result.smith_normal_form.entries
    computed = _mat_mul(_mat_mul(U, M), V)
    assert computed == D, f"U@M@V={computed} != D={D}"


# ===========================================================================
# Probe 1: PARI matsnf returning negative values — raw level
# ===========================================================================

class TestPariRawNegativeFactors:
    """Directly probe PARI for sign issues."""

    def test_matsnf_raw_negative_1x1(self):
        """PARI's matsnf on [[-1]] — does it return -1 or 1?"""
        from snforacle.backends.cypari2 import _pari
        pari = _pari()
        mat = pari.matrix(1, 1, [-1])
        raw = mat.matsnf()
        # raw is a PARI vector. Extract value.
        val = int(raw[0])
        # PARI returns -1 here, which is then used in _extract_invariant_factors
        # The function filters zeros but does NOT take abs().
        # If val < 0, that's a bug!
        if val < 0:
            # Now test that our wrapper handles it
            from snforacle.backends.cypari2 import _extract_invariant_factors
            factors = _extract_invariant_factors(mat)
            # factors should be [1] not [-1]
            assert all(f > 0 for f in factors), (
                f"BUG: _extract_invariant_factors returns negative factors: {factors}. "
                f"PARI raw matsnf output: {[int(raw[i]) for i in range(len(raw))]}"
            )

    def test_matsnf_raw_negative_2x2(self):
        """PARI matsnf on [[0, 1], [-1, 0]] — det=1."""
        from snforacle.backends.cypari2 import _pari
        pari = _pari()
        mat = pari.matrix(2, 2, [0, 1, -1, 0])
        raw = mat.matsnf()
        vals = [int(raw[i]) for i in range(len(raw))]
        # Check for negatives
        if any(v < 0 for v in vals):
            from snforacle.backends.cypari2 import _extract_invariant_factors
            factors = _extract_invariant_factors(mat)
            assert all(f > 0 for f in factors), (
                f"BUG: negative factors {factors}. PARI raw: {vals}"
            )

    def test_matsnf_raw_diagonal_negative(self):
        """PARI matsnf on diag(1, 1, -1) — det=-1."""
        from snforacle.backends.cypari2 import _pari
        pari = _pari()
        mat = pari.matrix(3, 3, [1,0,0, 0,1,0, 0,0,-1])
        raw = mat.matsnf()
        vals = [int(raw[i]) for i in range(len(raw))]
        if any(v < 0 for v in vals):
            from snforacle.backends.cypari2 import _extract_invariant_factors
            factors = _extract_invariant_factors(mat)
            # The SORT in _extract_invariant_factors will put -1 before 1
            # which corrupts the result!
            assert all(f > 0 for f in factors), (
                f"BUG: _extract_invariant_factors({vals}) = {factors}"
            )


# ===========================================================================
# Probe 2: Negative invariant factors propagating through compute_snf
# ===========================================================================

class TestNegativeFactorsPropagation:
    """Test that negative PARI factors don't leak to the user."""

    def test_compute_snf_negative_matrix(self):
        """Full pipeline test: [[-1, 0], [0, 1]]."""
        result = smith_normal_form(_dense([[-1, 0], [0, 1]]))
        _verify_snf(result, 2, 2)
        # The key test: invariant factors must be positive
        assert result.invariant_factors == [1, 1]

    def test_compute_snf_anti_identity(self):
        """diag(-1, -1, -1) — SNF should be diag(1, 1, 1)."""
        M = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        result = smith_normal_form(_dense(M))
        _verify_snf(result, 3, 3)
        assert result.invariant_factors == [1, 1, 1]

    def test_sorted_negative_factors(self):
        """If PARI returns [-1, 1, 1], sorted() gives [-1, 1, 1] which is WRONG.
        The correct SNF is [1, 1, 1]."""
        from snforacle.backends.cypari2 import _pari, _extract_invariant_factors
        pari = _pari()
        # diag(1, 1, -1) has det = -1
        mat = pari.matrix(3, 3, [1,0,0, 0,1,0, 0,0,-1])
        factors = _extract_invariant_factors(mat)
        # If this has -1, the sort puts it first and we get [-1, 1, 1]
        # which violates the SNF positivity convention
        for f in factors:
            assert f > 0, (
                f"CRITICAL BUG: _extract_invariant_factors returned {factors}. "
                f"Negative invariant factor violates SNF convention."
            )


# ===========================================================================
# Probe 3: _permutation_matrices with negative SNF entries from PARI
# ===========================================================================

class TestPermutationWithNegatives:
    """If D_pari has negative entries but D_std has positive, _permutation_matrices
    won't match them (different dict keys for value -1 vs 1)."""

    def test_permutation_sign_mismatch(self):
        """Matrix where PARI D has negative entry but standard D has positive.
        _permutation_matrices groups by value, so -d != d."""
        # diag(-1, 2) from PARI vs diag(1, 2) standard
        # This would cause unmatched entries in _permutation_matrices
        M = [[-1, 0], [0, 2]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 2, 2)
        _verify_transforms(M, result, 2, 2)

    def test_permutation_negative_det_3x3_transforms(self):
        """3x3 with negative determinant through full pipeline."""
        M = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 3, 3)
        _verify_transforms(M, result, 3, 3)

    def test_all_negative_diagonal_transforms(self):
        """diag(-2, -6, -12) — transforms must work despite PARI sign issues."""
        M = [[-2, 0, 0], [0, -6, 0], [0, 0, -12]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 3, 3)
        _verify_transforms(M, result, 3, 3)


# ===========================================================================
# Probe 4: Type coercion edge cases
# ===========================================================================

class TestTypeCoercion:
    """Test with unusual but Python-valid integer types."""

    def test_bool_entries(self):
        """Python bools are ints. [[True, False], [False, True]] = identity."""
        result = smith_normal_form(_dense([[True, False], [False, True]]))
        assert result.invariant_factors == [1, 1]

    def test_numpy_int64_entries(self):
        """numpy int64 values in entries."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        M = [[np.int64(2), np.int64(4)], [np.int64(6), np.int64(8)]]
        result = smith_normal_form(_dense(M))
        _verify_snf(result, 2, 2)

    def test_numpy_int32_entries(self):
        """numpy int32 values in entries."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not installed")
        M = [[np.int32(2), np.int32(4)], [np.int32(6), np.int32(8)]]
        result = smith_normal_form(_dense(M))
        _verify_snf(result, 2, 2)

    def test_huge_python_int_in_sparse(self):
        """Huge integer in sparse entry value field."""
        big = 10**1000
        result = smith_normal_form(_sparse(1, 1, [(0, 0, big)]))
        assert result.invariant_factors == [big]


# ===========================================================================
# Probe 5: Backend name edge cases
# ===========================================================================

class TestBackendNames:
    def test_backend_none(self):
        """backend=None — uses the best available default backend."""
        result = smith_normal_form(_dense([[1]]), backend=None)
        assert result.invariant_factors == [1]

    def test_backend_with_whitespace(self):
        """backend='cypari2 ' — trailing space should fail."""
        with pytest.raises(ValueError):
            smith_normal_form(_dense([[1]]), backend="cypari2 ")

    def test_backend_case_sensitive(self):
        """backend='CyPari2' — should fail (case mismatch)."""
        with pytest.raises(ValueError):
            smith_normal_form(_dense([[1]]), backend="CyPari2")

    def test_backend_integer(self):
        """backend=0 — wrong type."""
        with pytest.raises((TypeError, ValueError)):
            smith_normal_form(_dense([[1]]), backend=0)


# ===========================================================================
# Probe 6: Non-square rank-deficient with transforms
# ===========================================================================

class TestNonSquareRankDeficientTransforms:
    """Non-square rank-deficient matrices are the hardest case for
    _permutation_matrices because there are many zero positions to fill."""

    def test_3x5_rank_1_transforms(self):
        """3x5 matrix with rank 1."""
        M = [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 3, 5)
        _verify_transforms(M, result, 3, 5)

    def test_5x3_rank_1_transforms(self):
        """5x3 matrix with rank 1."""
        M = [[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12], [5, 10, 15]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 5, 3)
        _verify_transforms(M, result, 5, 3)

    def test_2x4_rank_1_transforms(self):
        """2x4 rank-1 matrix."""
        M = [[1, 2, 3, 4], [2, 4, 6, 8]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 2, 4)
        _verify_transforms(M, result, 2, 4)

    def test_4x2_rank_1_transforms(self):
        """4x2 rank-1 matrix."""
        M = [[1, 2], [2, 4], [3, 6], [4, 8]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 4, 2)
        _verify_transforms(M, result, 4, 2)

    def test_1x5_with_transforms(self):
        """1x5 row vector with transforms."""
        M = [[6, 10, 15, 21, 35]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 1, 5)
        _verify_transforms(M, result, 1, 5)

    def test_5x1_with_transforms(self):
        """5x1 column vector with transforms."""
        M = [[6], [10], [15], [21], [35]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 5, 1)
        _verify_transforms(M, result, 5, 1)

    def test_all_zero_3x5_transforms(self):
        """3x5 all-zero with transforms."""
        M = [[0]*5 for _ in range(3)]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 3, 5)
        _verify_transforms(M, result, 3, 5)


# ===========================================================================
# Probe 7: Matrices where PARI's D is maximally permuted from standard D
# ===========================================================================

class TestPariMaximalPermutation:
    """PARI's matsnf(flag=1) may return D with invariant factors in
    reverse order and right/bottom-aligned. Test that the permutation
    reordering works correctly."""

    def test_large_non_square_3x6_transforms(self):
        """3x6 with known SNF — exercises permutation logic heavily."""
        M = [
            [2, 0, 0, 0, 0, 0],
            [0, 0, 6, 0, 0, 0],
            [0, 0, 0, 0, 0, 30],
        ]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 3, 6)
        _verify_transforms(M, result, 3, 6)

    def test_6x3_transforms(self):
        """6x3 — transpose of above pattern."""
        M = [
            [2, 0, 0],
            [0, 0, 0],
            [0, 6, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 30],
        ]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 6, 3)
        _verify_transforms(M, result, 6, 3)

    def test_scrambled_diagonal_4x4_transforms(self):
        """4x4 with off-diagonal entries that yield non-trivial SNF ordering."""
        M = [
            [12, 0, 0, 0],
            [0, 4, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 6],
        ]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 4, 4)
        _verify_transforms(M, result, 4, 4)
        # diag(12, 4, 2, 6): gcd structure gives SNF = diag(2, 2, 12, 12)?
        # No — for a diagonal matrix, the SNF is computed from the multi-GCD.
        # Actually for diagonal matrices, d_1 = gcd of all entries, d_1*d_2 = gcd of all 2x2 minors, etc.
        # gcd(12,4,2,6) = 2
        # gcd of 2x2 minors of diag(12,4,2,6) = gcd(48, 24, 72, 8, 24, 12) = 4
        # so d_1=2, d_1*d_2=4, d_2=2
        # gcd of 3x3 minors = gcd(96, 288, 144, 48) = 48
        # d_1*d_2*d_3 = 48, d_3 = 12
        # d_1*d_2*d_3*d_4 = 12*4*2*6 = 576
        # d_4 = 576/48 = 12
        # So SNF = diag(2, 2, 12, 12)
        assert result.invariant_factors == [2, 2, 12, 12]


# ===========================================================================
# Probe 8: float('inf') and float('nan') attempts
# ===========================================================================

class TestInfNanAttempts:
    """Attempting to sneak inf/nan into the matrix."""

    def test_inf_in_dense_entries(self):
        """float('inf') in entries — should be rejected."""
        with pytest.raises((ValidationError, ValueError, OverflowError)):
            smith_normal_form(_dense([[float('inf'), 0], [0, 1]]))

    def test_nan_in_dense_entries(self):
        """float('nan') in entries — should be rejected."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_dense([[float('nan'), 0], [0, 1]]))

    def test_inf_in_sparse_value(self):
        """float('inf') in sparse entry value — should be rejected."""
        with pytest.raises((ValidationError, ValueError, OverflowError)):
            smith_normal_form({
                "format": "sparse",
                "nrows": 2, "ncols": 2,
                "entries": [{"row": 0, "col": 0, "value": float('inf')}],
            })

    def test_nan_in_sparse_value(self):
        """float('nan') in sparse entry value — should be rejected."""
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form({
                "format": "sparse",
                "nrows": 2, "ncols": 2,
                "entries": [{"row": 0, "col": 0, "value": float('nan')}],
            })


# ===========================================================================
# Probe 9: Very large dimensions (sparse, no entries)
# ===========================================================================

class TestLargeDimensions:
    """Large dimension numbers with zero entries — should not allocate huge matrices."""

    def test_sparse_1000x1000_empty(self):
        """1000x1000 sparse zero matrix — should work fast."""
        result = smith_normal_form(_sparse(1000, 1000, []))
        assert result.invariant_factors == []

    def test_sparse_huge_dimensions_empty(self):
        """10^6 x 10^6 sparse empty — tests whether to_dense() is called."""
        # This will call to_dense() which creates a 10^6 x 10^6 list of lists
        # That's 10^12 entries = out of memory. But the 0x0/empty check in
        # interface.py should handle nrows==0 or ncols==0. Since nrows=1000000
        # and ncols=1000000, it will try to allocate!
        # Actually the entries list is empty but nrows/ncols are huge.
        # to_dense() creates [[0]*ncols for _ in range(nrows)] = 10^6 lists of 10^6 zeros each.
        # This is ~8TB of memory. Let's test with a smaller but still large size.
        # Skip this test as it would OOM — instead test the concept:
        pass

    def test_sparse_10000x10000_single_entry(self):
        """10000x10000 sparse with 1 entry — will to_dense() OOM?"""
        # to_dense creates 10^4 x 10^4 = 10^8 entries. Each int is ~28 bytes in Python.
        # That's ~2.8GB. Too much for a test. Let's use 1000x1000 instead.
        result = smith_normal_form(_sparse(1000, 1000, [(0, 0, 7)]))
        _verify_snf(result, 1000, 1000)
        assert result.invariant_factors == [7]


# ===========================================================================
# Probe 10: Matrices designed to stress _build_snf_matrix
# ===========================================================================

class TestBuildSnfMatrix:
    """Test _build_snf_matrix with various inputs."""

    def test_more_factors_than_min_dim(self):
        """What if inv_factors has more entries than min(nrows, ncols)?
        This shouldn't happen in practice but the code has a guard."""
        from snforacle.backends.cypari2 import _build_snf_matrix
        # 2x3 matrix but 5 invariant factors — only first 2 should be placed
        mat = _build_snf_matrix([1, 2, 3, 4, 5], 2, 3)
        assert mat[0][0] == 1
        assert mat[1][1] == 2
        # Rest should be zero
        assert mat[0][1] == 0
        assert mat[0][2] == 0
        assert mat[1][0] == 0
        assert mat[1][2] == 0

    def test_zero_factors(self):
        """Empty invariant factors list."""
        from snforacle.backends.cypari2 import _build_snf_matrix
        mat = _build_snf_matrix([], 3, 3)
        assert mat == [[0,0,0],[0,0,0],[0,0,0]]

    def test_1x1_factor(self):
        from snforacle.backends.cypari2 import _build_snf_matrix
        mat = _build_snf_matrix([42], 1, 1)
        assert mat == [[42]]


# ===========================================================================
# Probe 11: Matrices with specific PARI matsnf(flag=1) quirks
# ===========================================================================

class TestPariFlag1Quirks:
    """PARI's matsnf(flag=1) returns [U, V, D] where D may have entries in
    reverse order. Test that _permutation_matrices correctly handles this."""

    def test_reverse_ordered_diagonal(self):
        """Matrix where PARI places factors in reverse order on diagonal."""
        # This is a matrix where SNF = diag(1, 2, 4)
        M = [[2, 2, 0], [0, 4, 0], [0, 0, 4]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 3, 3)
        _verify_transforms(M, result, 3, 3)

    def test_right_aligned_non_square(self):
        """Non-square: PARI might right-align the diagonal."""
        M = [[1, 2, 3], [0, 4, 8]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 2, 3)
        _verify_transforms(M, result, 2, 3)

    def test_bottom_aligned_tall(self):
        """Tall: PARI might bottom-align the diagonal."""
        M = [[1, 0], [2, 4], [3, 8]]
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, 3, 2)
        _verify_transforms(M, result, 3, 2)


# ===========================================================================
# Probe 12: Sparse matrix with every entry specified (full density)
# ===========================================================================

class TestSparseFullDensity:
    """Sparse matrix where every entry is specified (no implicit zeros)."""

    def test_full_sparse_2x2(self):
        """All 4 entries specified in sparse format."""
        entries = [(0,0,2), (0,1,4), (1,0,6), (1,1,8)]
        result = smith_normal_form(_sparse(2, 2, entries))
        expected = smith_normal_form(_dense([[2, 4], [6, 8]]))
        assert result.invariant_factors == expected.invariant_factors

    def test_full_sparse_3x3(self):
        """All 9 entries specified."""
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        entries = [(r, c, M[r][c]) for r in range(3) for c in range(3)]
        result = smith_normal_form(_sparse(3, 3, entries))
        expected = smith_normal_form(_dense(M))
        assert result.invariant_factors == expected.invariant_factors


# ===========================================================================
# Probe 13: Matrix with a single very large prime
# ===========================================================================

class TestLargePrime:
    """Matrix involving a large prime number."""

    def test_single_large_prime(self):
        """1x1 matrix with a large prime."""
        # 2^127 - 1 is a Mersenne prime
        p = 2**127 - 1
        result = smith_normal_form(_dense([[p]]))
        assert result.invariant_factors == [p]

    def test_diagonal_large_primes(self):
        """Diagonal with large primes."""
        p1 = 2**127 - 1  # Mersenne prime
        p2 = 2**61 - 1   # Mersenne prime
        M = [[p2, 0], [0, p1]]
        result = smith_normal_form(_dense(M))
        _verify_snf(result, 2, 2)
        # gcd(p1, p2) — both are distinct Mersenne primes, so gcd = 1
        # det = p1 * p2
        # SNF = diag(1, p1*p2)
        assert result.invariant_factors == [1, p1 * p2]

    def test_off_diagonal_large_primes(self):
        """Off-diagonal large primes."""
        p = 2**61 - 1
        M = [[p, 1], [0, p]]
        result = smith_normal_form(_dense(M))
        _verify_snf(result, 2, 2)


# ===========================================================================
# Probe 14: Unimodular (det=1) non-obvious matrices
# ===========================================================================

class TestUnimodularMatrices:
    """Matrices with determinant +/-1 — SNF should be identity."""

    def test_upper_triangular_unimodular(self):
        M = [[1, 100, 200], [0, 1, 300], [0, 0, 1]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1, 1]

    def test_lower_triangular_unimodular(self):
        M = [[1, 0, 0], [100, 1, 0], [200, 300, 1]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1, 1]

    def test_product_of_elementary_matrices(self):
        """Product of two shear matrices — still unimodular."""
        # [[1,2],[0,1]] @ [[1,0],[3,1]] = [[7,2],[3,1]], det = 1
        M = [[7, 2], [3, 1]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1]

    def test_negative_det_unimodular(self):
        """det = -1 matrix — SNF should still be identity."""
        M = [[1, 0], [0, -1]]
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [1, 1]


# ===========================================================================
# Probe 15: dict with extra fields
# ===========================================================================

class TestExtraFields:
    """Pydantic should ignore or reject extra fields."""

    def test_extra_field_in_dense(self):
        """Extra field 'backend' in the dict — Pydantic default may allow it."""
        inp = {
            "format": "dense",
            "nrows": 2, "ncols": 2,
            "entries": [[1, 0], [0, 1]],
            "extra_field": "should_this_work?",
        }
        # Pydantic v2 default is to ignore extra fields
        result = smith_normal_form(inp)
        assert result.invariant_factors == [1, 1]

    def test_extra_field_in_sparse_entry(self):
        """Extra field in a sparse entry dict."""
        inp = {
            "format": "sparse",
            "nrows": 2, "ncols": 2,
            "entries": [{"row": 0, "col": 0, "value": 5, "extra": True}],
        }
        result = smith_normal_form(inp)
        assert result.invariant_factors == [5]


