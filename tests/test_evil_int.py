"""Adversarial edge-case tests for all integer-matrix operations and backends.

Covers all five operations (SNF, SNF+transforms, HNF, HNF+transforms, elementary
divisors) across every available backend, plus schema validation, type coercion,
backend-name errors, sparse input, and large-integer behaviour.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from snforacle import (
    DenseIntMatrix,
    SparseIntMatrix,
    elementary_divisors,
    hermite_normal_form,
    hermite_normal_form_with_transform,
    smith_normal_form,
    smith_normal_form_with_transforms,
)


# ---------------------------------------------------------------------------
# Backend availability (evaluated once at import time)
# ---------------------------------------------------------------------------

def _avail(name: str) -> bool:
    """True only if the backend is installed and reachable."""
    import shutil
    if name == "pure_python":
        return True
    if name in ("sage", "magma"):
        if not shutil.which(name):
            return False
        try:
            smith_normal_form(
                {"format": "dense", "nrows": 1, "ncols": 1, "entries": [[1]]},
                backend=name,
            )
            return True
        except Exception:
            return False
    try:
        __import__(name)
        return True
    except ImportError:
        return False


_ALL          = ["cypari2", "flint", "sage", "magma", "pure_python"]
_ED_BACKENDS  = [b for b in _ALL if _avail(b)]
_HNF_BACKENDS = [b for b in _ALL if b != "cypari2" and _avail(b)]
_HNF_T_BACKENDS = [b for b in _ALL if b not in ("cypari2", "flint") and _avail(b)]
_SNF_BACKENDS = [b for b in _ALL if _avail(b)]
_SNF_T_BACKENDS = [b for b in _ALL if b != "flint" and _avail(b)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense(entries: list[list[int]], nrows: int, ncols: int) -> dict:
    return {"format": "dense", "nrows": nrows, "ncols": ncols, "entries": entries}


def _sparse(nrows: int, ncols: int, triples: list[tuple[int, int, int]] | None = None) -> dict:
    if triples is None:
        triples = []
    return {
        "format": "sparse",
        "nrows": nrows,
        "ncols": ncols,
        "entries": [{"row": r, "col": c, "value": v} for r, c, v in triples],
    }


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    m, n = len(A), len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _verify_hnf(H: list[list[int]], nrows: int, ncols: int) -> None:
    """Assert H is a valid row HNF (upper-trapezoidal, positive pivots)."""
    prev = -1
    for i in range(nrows):
        pj = next((j for j in range(ncols) if H[i][j] != 0), None)
        if pj is None:
            for k in range(i, nrows):
                assert all(H[k][j] == 0 for j in range(ncols))
            return
        assert pj > prev, f"pivot col {pj} ≤ previous {prev} at row {i}"
        assert H[i][pj] > 0, f"pivot H[{i}][{pj}] = {H[i][pj]} not positive"
        for j in range(pj):
            assert H[i][j] == 0
        for k in range(i + 1, nrows):
            assert H[k][pj] == 0
        prev = pj


# ---------------------------------------------------------------------------
# Test cases: (tag, nrows, ncols, entries, expected_inv_factors | None)
#
# expected_inv_factors = expected elementary divisors / SNF diagonal.
# None means "no hard-coded expectation; cross-backend consistency only".
# ---------------------------------------------------------------------------

_CASES = [
    # --- Trivial / degenerate ---
    ("zero_1x1",       1, 1, [[0]],                                              []),
    ("zero_3x3",       3, 3, [[0,0,0],[0,0,0],[0,0,0]],                         []),
    ("id_1x1",         1, 1, [[1]],                                              [1]),
    ("id_3x3",         3, 3, [[1,0,0],[0,1,0],[0,0,1]],                         [1,1,1]),

    # --- Negative entries (SNF always has positive invariant factors) ---
    ("neg_1x1",        1, 1, [[-1]],                                             [1]),
    ("neg_diag_2x2",   2, 2, [[-1,0],[0,-2]],                                    [1,2]),
    ("neg_det_2x2",    2, 2, [[-1,0],[0,2]],                                     [1,2]),
    ("all_neg_diag",   3, 3, [[-1,0,0],[0,-1,0],[0,0,-1]],                      [1,1,1]),

    # --- Standard textbook example ---
    ("std_3x3",        3, 3, [[2,4,4],[-6,6,12],[10,-4,-16]],                    [2,6,12]),

    # --- Diagonal matrices ---
    ("diag_2_4",       2, 2, [[2,0],[0,4]],                                      [2,4]),
    ("diag_6_4",       2, 2, [[6,0],[0,4]],                                      [2,12]),
    ("coprime_diag",   2, 2, [[7,0],[0,11]],                                     [1,77]),
    ("powers_of_2",    3, 3, [[2,0,0],[0,4,0],[0,0,8]],                         [2,4,8]),
    # Diagonal out of SNF order — tests that backends sort correctly
    ("diag_scramble",  4, 4, [[12,0,0,0],[0,4,0,0],[0,0,2,0],[0,0,0,6]],       [2,2,12,12]),

    # --- Rank-deficient square matrices ---
    ("rank1_3x3",      3, 3, [[1,2,3],[2,4,6],[3,6,9]],                         [1]),
    ("rank2_3x3",      3, 3, [[2,4,4],[-6,6,12],[4,8,8]],                       [2,6]),

    # --- Non-square matrices ---
    ("2x3",            2, 3, [[1,2,3],[4,5,6]],                                 [1,3]),
    ("3x2",            3, 2, [[1,4],[2,5],[3,6]],                               [1,3]),
    ("1x5_gcd3",       1, 5, [[3,6,9,12,15]],                                   [3]),
    ("5x1_gcd3",       5, 1, [[3],[6],[9],[12],[15]],                            [3]),
    ("3x4_arith",      3, 4, [[1,2,3,4],[5,6,7,8],[9,10,11,12]],                [1,4]),
    ("5x2_rank2",      5, 2, [[1,0],[0,1],[0,0],[0,0],[0,0]],                   [1,1]),
    ("2x5_zero",       2, 5, [[0,0,0,0,0],[0,0,0,0,0]],                         []),
    # Wide rank-1 matrices (stress _permutation_matrices for non-square transforms)
    ("3x5_rank1",      3, 5, [[1,2,3,4,5],[2,4,6,8,10],[3,6,9,12,15]],         [1]),
    ("5x3_rank1",      5, 3, [[1,2,3],[2,4,6],[3,6,9],[4,8,12],[5,10,15]],      [1]),
    # Vectors with gcd=1 (exercises non-trivial Bezout computation)
    ("1x5_gcd1",       1, 5, [[6,10,15,21,35]],                                 [1]),
    ("5x1_gcd1",       5, 1, [[6],[10],[15],[21],[35]],                          [1]),

    # --- Small determinant ---
    ("det2_2x2",       2, 2, [[1,2],[3,4]],                                      [1,2]),

    # --- Unimodular (det = ±1, SNF = identity) ---
    ("unimod_3x3",     3, 3, [[1,2,3],[0,1,4],[0,0,1]],                         [1,1,1]),
    ("unimod_shear",   2, 2, [[7,2],[3,1]],                                      [1,1]),
    ("unimod_neg_det", 2, 2, [[1,0],[0,-1]],                                     [1,1]),

    # --- Large integers ---
    ("mersenne_61",    1, 1, [[2**61-1]],                                        [2**61-1]),
    ("large_prime",    1, 1, [[2**127-1]],                                       [2**127-1]),
    ("two_mersennes",  2, 2, [[2**61-1,0],[0,2**127-1]],                        [1,(2**61-1)*(2**127-1)]),
]

_IDS = [c[0] for c in _CASES]


# ---------------------------------------------------------------------------
# 1. Elementary divisors
# ---------------------------------------------------------------------------

class TestEvilED:
    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_ed(self, tag, nrows, ncols, entries, expected_inv):
        assert _ED_BACKENDS, "No ED backends available"
        inp = _dense(entries, nrows, ncols)
        ref_b = _ED_BACKENDS[0]
        ref = elementary_divisors(inp, backend=ref_b)

        if expected_inv is not None:
            assert ref.elementary_divisors == expected_inv, (
                f"{tag}: expected {expected_inv}, {ref_b} returned {ref.elementary_divisors}"
            )

        for b in _ED_BACKENDS[1:]:
            result = elementary_divisors(inp, backend=b)
            assert result.elementary_divisors == ref.elementary_divisors, (
                f"{tag}: {ref_b}={ref.elementary_divisors} vs {b}={result.elementary_divisors}"
            )


# ---------------------------------------------------------------------------
# 2. Hermite Normal Form
# ---------------------------------------------------------------------------

class TestEvilHNF:
    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_hnf(self, tag, nrows, ncols, entries, expected_inv):
        assert _HNF_BACKENDS, "No HNF backends available"
        inp = _dense(entries, nrows, ncols)
        ref_b = _HNF_BACKENDS[0]
        ref = hermite_normal_form(inp, backend=ref_b)
        H_ref = ref.hermite_normal_form.entries

        _verify_hnf(H_ref, nrows, ncols)

        for b in _HNF_BACKENDS[1:]:
            result = hermite_normal_form(inp, backend=b)
            H = result.hermite_normal_form.entries
            assert H == H_ref, (
                f"{tag}: {ref_b} vs {b} HNF differ:\n  {ref_b}: {H_ref}\n  {b}: {H}"
            )


# ---------------------------------------------------------------------------
# 3. HNF with transform
# ---------------------------------------------------------------------------

class TestEvilHNFWithTransform:
    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_hnf_transform(self, tag, nrows, ncols, entries, expected_inv):
        if not _HNF_T_BACKENDS:
            pytest.skip("No HNF-transform backends available")
        inp = _dense(entries, nrows, ncols)
        ref_hnf = (
            hermite_normal_form(inp, backend=_HNF_BACKENDS[0]).hermite_normal_form.entries
            if _HNF_BACKENDS else None
        )
        for b in _HNF_T_BACKENDS:
            result = hermite_normal_form_with_transform(inp, backend=b)
            H = result.hermite_normal_form.entries
            U = result.left_transform.entries
            _verify_hnf(H, nrows, ncols)
            assert _mat_mul(U, entries) == H, f"{tag}: U@M ≠ H for {b}"
            if ref_hnf is not None:
                assert H == ref_hnf, f"{tag}: {b} HNF differs from {_HNF_BACKENDS[0]}"


# ---------------------------------------------------------------------------
# 4. Smith Normal Form
# ---------------------------------------------------------------------------

class TestEvilSNF:
    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_snf(self, tag, nrows, ncols, entries, expected_inv):
        assert _SNF_BACKENDS, "No SNF backends available"
        inp = _dense(entries, nrows, ncols)
        ref_b = _SNF_BACKENDS[0]
        ref = smith_normal_form(inp, backend=ref_b)

        if expected_inv is not None:
            assert ref.invariant_factors == expected_inv, (
                f"{tag}: expected {expected_inv}, {ref_b} returned {ref.invariant_factors}"
            )

        for b in _SNF_BACKENDS[1:]:
            result = smith_normal_form(inp, backend=b)
            assert result.invariant_factors == ref.invariant_factors, (
                f"{tag}: {ref_b}={ref.invariant_factors} vs {b}={result.invariant_factors}"
            )
            assert result.smith_normal_form.entries == ref.smith_normal_form.entries, (
                f"{tag}: {ref_b} vs {b} SNF matrix entries differ"
            )


# ---------------------------------------------------------------------------
# 5. SNF with transforms
# ---------------------------------------------------------------------------

class TestEvilSNFWithTransforms:
    @pytest.mark.parametrize("tag,nrows,ncols,entries,expected_inv", _CASES, ids=_IDS)
    def test_snf_transforms(self, tag, nrows, ncols, entries, expected_inv):
        assert _SNF_T_BACKENDS, "No SNF-transform backends available"
        inp = _dense(entries, nrows, ncols)
        ref_inv = (
            smith_normal_form(inp, backend=_SNF_BACKENDS[0]).invariant_factors
            if _SNF_BACKENDS else None
        )
        for b in _SNF_T_BACKENDS:
            result = smith_normal_form_with_transforms(inp, backend=b)
            if ref_inv is not None:
                assert result.invariant_factors == ref_inv, (
                    f"{tag}: {b} factors={result.invariant_factors} ≠ ref={ref_inv}"
                )
            U = result.left_transform.entries
            V = result.right_transform.entries
            S = result.smith_normal_form.entries
            assert _mat_mul(_mat_mul(U, entries), V) == S, (
                f"{tag}: U@M@V ≠ SNF for {b}"
            )


# ---------------------------------------------------------------------------
# 6. Sparse input
# ---------------------------------------------------------------------------

class TestSparseInput:
    """Sparse-format inputs should give identical results to their dense equivalents."""

    def test_sparse_zero_1000x1000(self):
        result = smith_normal_form(_sparse(1000, 1000, []))
        assert result.invariant_factors == []

    def test_sparse_single_entry_1000x1000(self):
        result = smith_normal_form(_sparse(1000, 1000, [(0, 0, 7)]))
        assert result.invariant_factors == [7]

    def test_sparse_vs_dense_consistency(self):
        M = [[2, 4, 4], [-6, 6, 12], [10, -4, -16]]
        triples = [(r, c, M[r][c]) for r in range(3) for c in range(3) if M[r][c] != 0]
        r_dense = smith_normal_form(_dense(M, 3, 3))
        r_sparse = smith_normal_form(_sparse(3, 3, triples))
        assert r_dense.invariant_factors == r_sparse.invariant_factors

    def test_sparse_pydantic_model_input(self):
        from snforacle.schema import SparseEntry
        m = SparseIntMatrix(
            format="sparse", nrows=3, ncols=3,
            entries=[SparseEntry(row=0, col=0, value=2),
                     SparseEntry(row=1, col=1, value=6),
                     SparseEntry(row=2, col=2, value=12)],
        )
        result = elementary_divisors(m)
        assert result.elementary_divisors == [2, 6, 12]


# ---------------------------------------------------------------------------
# 7. Schema / input validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Malformed inputs must be rejected before reaching any backend."""

    def test_wrong_nrows(self):
        with pytest.raises(ValidationError):
            smith_normal_form({"format": "dense", "nrows": 3, "ncols": 2,
                               "entries": [[1, 2], [3, 4]]})

    def test_wrong_ncols(self):
        with pytest.raises(ValidationError):
            smith_normal_form({"format": "dense", "nrows": 2, "ncols": 3,
                               "entries": [[1, 2], [3, 4]]})

    def test_sparse_out_of_bounds_row(self):
        with pytest.raises(ValidationError):
            smith_normal_form(_sparse(2, 2, [(5, 0, 1)]))

    def test_sparse_out_of_bounds_col(self):
        with pytest.raises(ValidationError):
            smith_normal_form(_sparse(2, 2, [(0, 5, 1)]))

    def test_sparse_duplicate_position(self):
        with pytest.raises(ValidationError):
            smith_normal_form(_sparse(2, 2, [(0, 0, 1), (0, 0, 2)]))

    def test_sparse_zero_value_rejected(self):
        with pytest.raises(ValidationError):
            smith_normal_form(_sparse(2, 2, [(0, 0, 0)]))

    def test_inf_in_dense_entries(self):
        with pytest.raises((ValidationError, ValueError, OverflowError)):
            smith_normal_form(_dense([[float("inf"), 0], [0, 1]], 2, 2))

    def test_nan_in_dense_entries(self):
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_dense([[float("nan"), 0], [0, 1]], 2, 2))

    def test_string_in_entries(self):
        with pytest.raises((ValidationError, ValueError)):
            smith_normal_form(_dense([["a", 0], [0, 1]], 2, 2))

    def test_extra_fields_in_dense_ignored(self):
        """Pydantic v2 ignores unknown fields by default."""
        result = smith_normal_form({
            "format": "dense", "nrows": 1, "ncols": 1,
            "entries": [[5]], "extra": "ignored",
        })
        assert result.invariant_factors == [5]

    def test_huge_sparse_dimensions_rejected(self):
        """_MAX_DENSE_ELEMENTS guard must reject a sparse matrix whose dense
        expansion would be unreasonably large."""
        with pytest.raises((ValueError, MemoryError)):
            smith_normal_form(_sparse(100_000, 100_000, [(0, 0, 1)]))


# ---------------------------------------------------------------------------
# 8. Type coercion
# ---------------------------------------------------------------------------

class TestTypeCoercion:
    """Python bools and numpy integers are valid integer types."""

    def test_bool_entries(self):
        result = smith_normal_form(_dense([[True, False], [False, True]], 2, 2))
        assert result.invariant_factors == [1, 1]

    def test_numpy_int64(self):
        np = pytest.importorskip("numpy")
        M = [[np.int64(2), np.int64(4)], [np.int64(6), np.int64(8)]]
        result = smith_normal_form(_dense(M, 2, 2))
        assert result.invariant_factors == [2, 4]

    def test_numpy_int32(self):
        np = pytest.importorskip("numpy")
        M = [[np.int32(1), np.int32(0)], [np.int32(0), np.int32(1)]]
        result = smith_normal_form(_dense(M, 2, 2))
        assert result.invariant_factors == [1, 1]

    def test_huge_python_int_sparse(self):
        big = 10**1000
        result = smith_normal_form(_sparse(1, 1, [(0, 0, big)]))
        assert result.invariant_factors == [big]


# ---------------------------------------------------------------------------
# 9. Backend name errors
# ---------------------------------------------------------------------------

class TestBackendNames:
    def test_backend_none_uses_default(self):
        result = smith_normal_form(_dense([[1]], 1, 1), backend=None)
        assert result.invariant_factors == [1]

    def test_backend_trailing_space_rejected(self):
        with pytest.raises(ValueError):
            smith_normal_form(_dense([[1]], 1, 1), backend="cypari2 ")

    def test_backend_wrong_case_rejected(self):
        with pytest.raises(ValueError):
            smith_normal_form(_dense([[1]], 1, 1), backend="CyPari2")

    def test_backend_unknown_name_rejected(self):
        with pytest.raises(ValueError):
            smith_normal_form(_dense([[1]], 1, 1), backend="nonexistent")

    def test_backend_integer_type_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            smith_normal_form(_dense([[1]], 1, 1), backend=0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. Pydantic model round-trip
# ---------------------------------------------------------------------------

class TestPydanticRoundtrip:
    def test_dense_model_input(self):
        m = DenseIntMatrix(format="dense", nrows=2, ncols=2, entries=[[1, 0], [0, 1]])
        result = smith_normal_form(m)
        assert result.invariant_factors == [1, 1]

    def test_model_dump_round_trip(self):
        m = DenseIntMatrix(format="dense", nrows=2, ncols=2, entries=[[3, 0], [0, 7]])
        result = smith_normal_form(m.model_dump())
        assert result.invariant_factors == [1, 21]

    def test_json_string_rejected(self):
        import json
        s = json.dumps({"format": "dense", "nrows": 1, "ncols": 1, "entries": [[1]]})
        with pytest.raises(Exception):
            smith_normal_form(s)  # type: ignore[arg-type]

    def test_result_model_dump(self):
        result = smith_normal_form(_dense([[2, 0], [0, 6]], 2, 2))
        d = result.model_dump()
        assert d["invariant_factors"] == [2, 6]

    def test_result_model_dump_json(self):
        import json
        result = elementary_divisors(_dense([[2, 0], [0, 6]], 2, 2))
        d = json.loads(result.model_dump_json())
        assert d["elementary_divisors"] == [2, 6]
