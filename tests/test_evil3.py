"""Adversarial test suite, round 3 — exploiting confirmed vulnerabilities.

Focus: 
1. PARI stack overflow for large sparse matrices (confirmed bug)
2. Memory exhaustion from sparse to_dense() conversion (design issue)
3. Race conditions in _pari singleton
4. Matrices that cause PARI to produce unexpected D structures
5. Edge cases in the Q permutation matrix construction (transpose logic)
"""

import pytest
import sys
import os

from snforacle import smith_normal_form, smith_normal_form_with_transforms
from snforacle.schema import SNFResult


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
# BUG #1: PARI stack overflow on large sparse matrices
# ===========================================================================

class TestPariStackOverflow:
    """PARI's stack has been increased to 128 MB (see _pari() in cypari2.py),
    so 1000x1000 matrices now work. These tests verify that PARI can handle
    large sparse matrices after the stack increase."""

    def test_sparse_1000x1000_zero_cypari2(self):
        """1000x1000 sparse zero matrix now works with enlarged PARI stack."""
        result = smith_normal_form(_sparse(1000, 1000, []), backend="cypari2")
        assert result.invariant_factors == []

    def test_sparse_1000x1000_single_entry_cypari2(self):
        """1000x1000 sparse with 1 entry now works with enlarged PARI stack."""
        result = smith_normal_form(_sparse(1000, 1000, [(0, 0, 7)]), backend="cypari2")
        assert result.invariant_factors == [7]

    def test_sparse_1000x1000_zero_flint(self):
        """1000x1000 via flint — may or may not work (depends on flint's stack)."""
        # Let's see if flint handles it
        result = smith_normal_form(_sparse(1000, 1000, []), backend="flint")
        assert result.invariant_factors == []

    def test_sparse_1000x1000_single_entry_flint(self):
        """1000x1000 sparse single entry via flint."""
        result = smith_normal_form(
            _sparse(1000, 1000, [(0, 0, 7)]), backend="flint"
        )
        assert result.invariant_factors == [7]


# ===========================================================================
# BUG #2: Sparse to_dense() memory explosion for large dimensions
# ===========================================================================

class TestSparseToDenseMemoryIssue:
    """SparseIntMatrix.to_dense() creates [[0]*ncols for _ in range(nrows)]
    regardless of the number of actual entries. For nrows=ncols=10000,
    this is 10^8 Python int objects (~2.8 GB)."""

    def test_sparse_to_dense_allocates_full_matrix(self):
        """Verify that to_dense() actually allocates the full matrix."""
        from snforacle.schema import SparseIntMatrix, SparseEntry
        m = SparseIntMatrix(
            format="sparse", nrows=100, ncols=100, entries=[]
        )
        dense = m.to_dense()
        assert len(dense) == 100
        assert len(dense[0]) == 100
        # All zeros
        assert all(dense[i][j] == 0 for i in range(100) for j in range(100))

    def test_interface_always_calls_to_dense(self):
        """The interface always calls to_dense() even for sparse matrices.
        This means there's no memory savings from using sparse format."""
        # For moderate sizes this is fine, but for large dimensions it's a problem.
        # Test with a moderate but non-trivial size
        result = smith_normal_form(_sparse(200, 200, [(0, 0, 42)]))
        assert result.invariant_factors == [42]


# ===========================================================================
# Probe: What about nrows or ncols as very large ints in sparse?
# ===========================================================================

class TestSparseWithHugeDimensions:
    """Sparse matrix with huge nrows/ncols but no entries.
    Pydantic allows it (ge=0), but to_dense() will try to allocate."""

    def test_huge_nrows_sparse_empty_validation_passes(self):
        """Pydantic validation accepts nrows=10^9 with no entries."""
        from snforacle.schema import SparseIntMatrix
        # This should pass validation — the validator only checks bounds
        m = SparseIntMatrix(
            format="sparse", nrows=1_000_000_000, ncols=1_000_000_000,
            entries=[]
        )
        assert m.nrows == 1_000_000_000
        # But if we call smith_normal_form, the nrows/ncols check
        # won't catch it (nrows != 0 and ncols != 0) so it'll call to_dense()
        # which will try to allocate 10^18 entries = OOM crash

    def test_huge_sparse_would_oom_if_not_caught(self):
        """Document that calling smith_normal_form with huge sparse matrix
        will attempt to allocate nrows*ncols memory."""
        # We don't actually run this — it would OOM-kill the process
        # But we document the issue
        pass


# ===========================================================================
# Probe: Q matrix construction in _permutation_matrices
# ===========================================================================

class TestQMatrixConstruction:
    """The Q matrix in _permutation_matrices uses a transposed convention:
    Q[j_p][j_s] = 1 (line 118) vs P[i_s][i_p] = 1 (line 114).
    
    This asymmetry is intentional: P is a left-multiply matrix (row permutation)
    and Q is a right-multiply matrix (column permutation).
    For P @ D @ Q = D_std:
    - P maps pari rows to std rows: (P @ D)[i_s] = D[i_p]
    - Q maps std cols to pari cols: (D @ Q)[j_p] = D[j_s]
    
    Wait — let's verify this is correct.
    """

    def test_q_matrix_convention(self):
        """Verify Q matrix convention is correct."""
        from snforacle.backends.cypari2 import _permutation_matrices, _mat_mul
        
        # D_pari = [[0,5],[3,0]] — entries at (0,1)=5 and (1,0)=3
        # D_std = [[3,0],[0,5]] — entries at (0,0)=3 and (1,1)=5
        D_pari = [[0, 5], [3, 0]]
        D_std = [[3, 0], [0, 5]]
        
        P, Q = _permutation_matrices(D_pari, D_std, 2, 2)
        
        result = _mat_mul(_mat_mul(P, D_pari), Q)
        assert result == D_std, f"P @ D_pari @ Q = {result} != {D_std}"

    def test_q_matrix_non_square(self):
        """Non-square Q matrix convention."""
        from snforacle.backends.cypari2 import _permutation_matrices, _mat_mul
        
        # D_pari = [[0,0,7],[0,0,0]] — entry at (0,2)=7
        # D_std = [[7,0,0],[0,0,0]] — entry at (0,0)=7
        D_pari = [[0, 0, 7], [0, 0, 0]]
        D_std = [[7, 0, 0], [0, 0, 0]]
        
        P, Q = _permutation_matrices(D_pari, D_std, 2, 3)
        result = _mat_mul(_mat_mul(P, D_pari), Q)
        assert result == D_std, f"P @ D_pari @ Q = {result} != {D_std}"


# ===========================================================================
# Probe: Random matrix transforms (stress test)
# ===========================================================================

class TestRandomMatrixTransforms:
    """Use deterministic 'random' matrices to stress test transforms."""

    def _make_matrix(self, nrows, ncols, seed=42):
        """Deterministic pseudo-random matrix."""
        import random
        rng = random.Random(seed)
        return [[rng.randint(-100, 100) for _ in range(ncols)] for _ in range(nrows)]

    @pytest.mark.parametrize("nrows,ncols", [
        (2, 2), (3, 3), (4, 4), (5, 5),
        (2, 3), (3, 2), (2, 5), (5, 2),
        (1, 4), (4, 1), (3, 6), (6, 3),
        (1, 1), (7, 7), (8, 3), (3, 8),
    ])
    def test_random_transforms(self, nrows, ncols):
        """Transform verification on deterministic random matrices."""
        M = self._make_matrix(nrows, ncols, seed=nrows * 100 + ncols)
        result = smith_normal_form_with_transforms(_dense(M))
        _verify_snf(result, nrows, ncols)
        _verify_transforms(M, result, nrows, ncols)

    @pytest.mark.parametrize("nrows,ncols", [
        (2, 2), (3, 3), (4, 4), (5, 5),
        (2, 3), (3, 2), (2, 5), (5, 2),
        (1, 4), (4, 1), (3, 6), (6, 3),
    ])
    def test_random_cross_backend(self, nrows, ncols):
        """Cross-backend consistency on random matrices."""
        M = self._make_matrix(nrows, ncols, seed=nrows * 100 + ncols)
        r_pari = smith_normal_form(_dense(M), backend="cypari2")
        r_flint = smith_normal_form(_dense(M), backend="flint")
        assert r_pari.invariant_factors == r_flint.invariant_factors, (
            f"{nrows}x{ncols}: cypari2={r_pari.invariant_factors} "
            f"vs flint={r_flint.invariant_factors}"
        )


# ===========================================================================
# Probe: Matrices with very specific GCD structure
# ===========================================================================

class TestGCDStructure:
    """Matrices designed to test the GCD/divisibility computation."""

    def test_powers_of_2(self):
        """diag(2, 4, 8, 16, 32) — all powers of 2."""
        n = 5
        M = [[0]*n for _ in range(n)]
        for i in range(n):
            M[i][i] = 2**(i+1)
        result = smith_normal_form(_dense(M))
        assert result.invariant_factors == [2, 4, 8, 16, 32]

    def test_highly_composite(self):
        """Matrix with highly composite numbers."""
        M = [[120, 0], [0, 360]]
        result = smith_normal_form(_dense(M))
        _verify_snf(result, 2, 2)
        # gcd(120, 360) = 120, lcm(120, 360) = 360
        # d1 = gcd(120, 360) = 120? No...
        # For diagonal matrix: d1 = gcd(all entries) = gcd(120, 360) = 120
        # d1 * d2 = det = 120 * 360 = 43200
        # d2 = 43200 / 120 = 360
        assert result.invariant_factors == [120, 360]

    def test_coprime_diagonal(self):
        """diag(p1, p2) where p1, p2 are coprime primes."""
        M = [[7, 0], [0, 11]]
        result = smith_normal_form(_dense(M))
        # gcd(7, 11) = 1, so d1=1, d2=77
        assert result.invariant_factors == [1, 77]

    def test_same_prime_different_powers(self):
        """diag(p^a, p^b) where a < b."""
        M = [[8, 0], [0, 32]]  # 2^3, 2^5
        result = smith_normal_form(_dense(M))
        # gcd(8, 32) = 8, lcm = 32
        assert result.invariant_factors == [8, 32]


# ===========================================================================
# Probe: Interface _parse_matrix edge cases
# ===========================================================================

class TestParseMatrixEdgeCases:
    """Test the _parse_matrix function in interface.py."""

    def test_dict_with_model_instance(self):
        """Passing a DenseIntMatrix instance directly."""
        from snforacle.schema import DenseIntMatrix
        m = DenseIntMatrix(format="dense", nrows=2, ncols=2, entries=[[1,0],[0,1]])
        result = smith_normal_form(m)
        assert result.invariant_factors == [1, 1]

    def test_dict_round_trip(self):
        """model_dump() of a DenseIntMatrix should be re-parseable."""
        from snforacle.schema import DenseIntMatrix
        m = DenseIntMatrix(format="dense", nrows=2, ncols=2, entries=[[3,0],[0,7]])
        d = m.model_dump()
        result = smith_normal_form(d)
        assert result.invariant_factors == [1, 21]

    def test_json_string_input(self):
        """Passing a JSON string (not a dict) — should this work?"""
        import json
        d = {"format": "dense", "nrows": 1, "ncols": 1, "entries": [[42]]}
        json_str = json.dumps(d)
        # _parse_matrix calls validate_python, not validate_json
        with pytest.raises((Exception,)):
            smith_normal_form(json_str)


# ===========================================================================
# Probe: Flint fmpz_mat.snf() edge cases
# ===========================================================================

class TestFlintEdgeCases:
    """Specific tests for the flint backend."""

    def test_flint_1x1_zero(self):
        result = smith_normal_form(_dense([[0]]), backend="flint")
        assert result.invariant_factors == []

    def test_flint_1x1_negative(self):
        result = smith_normal_form(_dense([[-1]]), backend="flint")
        assert result.invariant_factors == [1]

    def test_flint_non_square_3x2(self):
        M = [[1, 4], [2, 5], [3, 6]]
        result = smith_normal_form(_dense(M), backend="flint")
        _verify_snf(result, 3, 2)

    def test_flint_large_integers(self):
        big = 2**500
        M = [[big, 0], [0, big]]
        result = smith_normal_form(_dense(M), backend="flint")
        assert result.invariant_factors == [big, big]

    def test_flint_rank_0(self):
        M = [[0, 0, 0], [0, 0, 0]]
        result = smith_normal_form(_dense(M), backend="flint")
        assert result.invariant_factors == []


# ===========================================================================
# Probe: Matrices that might cause _permutation_matrices to fail
# ===========================================================================

class TestPermutationMatricesFailureModes:
    """Try to make _permutation_matrices produce wrong results."""

    def test_d_pari_has_more_nonzero_than_d_std(self):
        """Can D_pari have more nonzero entries than D_std?
        No — they should have the same number of nonzero entries
        (same rank). But what if they differ?"""
        # This can't happen in practice because both are computed from 
        # the same matrix. But _permutation_matrices doesn't validate this.
        from snforacle.backends.cypari2 import _permutation_matrices
        
        # Adversarial: D_pari has 2 nonzero, D_std has 1
        # This should not happen in practice but let's see what breaks
        D_pari = [[3, 0], [0, 5]]
        D_std = [[3, 0], [0, 0]]
        
        # This will have mismatched values — the 5 in D_pari has no match in D_std
        # The function will silently leave the 5's position unhandled
        try:
            P, Q = _permutation_matrices(D_pari, D_std, 2, 2)
            from snforacle.backends.cypari2 import _mat_mul
            result = _mat_mul(_mat_mul(P, D_pari), Q)
            # This likely won't match D_std
            if result != D_std:
                pass  # Expected — adversarial input
        except (KeyError, IndexError) as e:
            pass  # Also acceptable

    def test_d_pari_and_d_std_same_values_different_multiplicities(self):
        """D_pari = diag(1, 1, 2) but D_std = diag(1, 2, 2)."""
        from snforacle.backends.cypari2 import _permutation_matrices, _mat_mul
        
        D_pari = [[1,0,0],[0,1,0],[0,0,2]]
        D_std = [[1,0,0],[0,2,0],[0,0,2]]
        
        # zip will silently truncate the longer list
        try:
            P, Q = _permutation_matrices(D_pari, D_std, 3, 3)
            result = _mat_mul(_mat_mul(P, D_pari), Q)
            # This will NOT match D_std because the values are different
        except Exception:
            pass


# ===========================================================================
# Probe: What if backend raises unexpected exception?
# ===========================================================================

class TestBackendExceptionHandling:
    """Test error handling when backends fail."""

    def test_cypari2_import_error_message(self):
        """If cypari2 is not installed, the error message should be helpful."""
        # We can't test this directly since cypari2 IS installed.
        # But we can verify the error path exists.
        from snforacle.backends.cypari2 import _pari
        # Just verify it doesn't crash on import
        pari = _pari()
        assert pari is not None

    def test_flint_import_error_message(self):
        """Same for flint."""
        from snforacle.backends.flint import _flint
        flint = _flint()
        assert flint is not None


