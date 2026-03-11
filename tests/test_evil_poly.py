"""Evil edge-case tests for polynomial matrix operations over F_p[x].

Tests all 5 operations across all available poly backends on a set of
mathematically challenging matrices (rank-deficient, near-singular, high
degree, non-square, all-zero rows/cols, repeated factors, etc.).
"""

from __future__ import annotations

import shutil

import pytest

from snforacle import (
    poly_elementary_divisors,
    poly_hermite_normal_form,
    poly_hermite_normal_form_with_transform,
    poly_smith_normal_form,
    poly_smith_normal_form_with_transforms,
)
from snforacle.backends.pure_python_poly import poly_mat_mul
from _mathelpers import assert_invertible_poly

# ---------------------------------------------------------------------------
# Available backends
# ---------------------------------------------------------------------------


def _avail(name: str) -> bool:
    if name == "pure_python":
        return True
    return shutil.which(name) is not None


_ALL_BACKENDS = [b for b in ["pure_python", "sage", "magma"] if _avail(b)]
_SNF_BACKENDS = _ALL_BACKENDS
_HNF_BACKENDS = _ALL_BACKENDS
_SNF_T_BACKENDS = _ALL_BACKENDS
_HNF_T_BACKENDS = _ALL_BACKENDS
_ED_BACKENDS = _ALL_BACKENDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense(entries: list[list[list[int]]], p: int) -> dict:
    return {
        "format": "dense_poly",
        "nrows": len(entries),
        "ncols": len(entries[0]) if entries else 0,
        "p": p,
        "entries": entries,
    }


def _poly_eq(a, b) -> bool:
    def trim(x):
        c = list(x)
        while c and c[-1] == 0:
            c.pop()
        return c
    return trim(a) == trim(b)


def _mat_eq(A, B) -> bool:
    if len(A) != len(B):
        return False
    for ra, rb in zip(A, B):
        if len(ra) != len(rb):
            return False
        for pa, pb in zip(ra, rb):
            if not _poly_eq(pa, pb):
                return False
    return True


def _is_upper_triangular(H) -> bool:
    ncols = len(H[0]) if H else 0
    for i in range(len(H)):
        for j in range(min(i, ncols)):
            if H[i][j]:
                return False
    return True


def _is_monic_or_zero(poly) -> bool:
    if not poly:
        return True
    return poly[-1] == 1


# ---------------------------------------------------------------------------
# Evil edge cases
# ---------------------------------------------------------------------------

# Format: (label, entries, p)
_CASES: list[tuple[str, list[list[list[int]]], int]] = [
    # 1. 1×1 zero
    ("1x1_zero", [[[]]], 2),
    # 2. 1×1 constant unit
    ("1x1_unit_f7", [[[3]]], 7),
    # 3. 1×1 polynomial x over F_2
    ("1x1_x_f2", [[[0, 1]]], 2),
    # 4. 2×2 zero over F_3
    ("2x2_zero_f3", [[[], []], [[], []]], 3),
    # 5. 2×2 identity over F_2
    ("2x2_identity_f2", [[[1], []], [[], [1]]], 2),
    # 6. 2×2 [[x, 0], [0, x]] — both diagonal entries same
    ("2x2_diag_xx_f5", [[[0, 1], []], [[], [0, 1]]], 5),
    # 7. 2×2 [[x, 1], [0, x]] — upper triangular, x | x (trivially)
    ("2x2_upper_tri_f2", [[[0, 1], [1]], [[], [0, 1]]], 2),
    # 8. 2×2 rank 1: [[x, x^2], [1, x]] — second row = x * first
    #    Actually rank 2; let's do [[x, x^2], [2x, 2x^2]] mod 3 rank 1
    ("2x2_rank1_f3", [[[0, 1], [0, 0, 1]], [[0, 2], [0, 0, 2]]], 3),
    # 9. 2×2 [[x+1, x^2+x], [1, x]] — factor out (x+1)
    ("2x2_common_factor_f2", [[[1, 1], [0, 1, 1]], [[1], [0, 1]]], 2),
    # 10. 3×3 diagonal with gcd-chain: diag(1, x, x^2)
    ("3x3_diag_1_x_x2_f5", [[[1], [], []], [[], [0, 1], []], [[], [], [0, 0, 1]]], 5),
    # 11. 3×3 diagonal that violates divisibility: diag(x, 1, x^2) → SNF reorders
    ("3x3_diag_x_1_x2_f5", [[[0, 1], [], []], [[], [1], []], [[], [], [0, 0, 1]]], 5),
    # 12. 3×3 with all-zero first row
    ("3x3_zero_first_row_f2", [[[], [], []], [[1], [0, 1], [1, 1]], [[0, 1], [1], []]], 2),
    # 13. 3×3 with all-zero last column
    ("3x3_zero_last_col_f7", [[[1], [0, 1], []], [[3], [2, 1], []], [[0, 1], [4], []]], 7),
    # 14. Non-square 1×3 row vector
    ("1x3_row_vector_f2", [[[0, 1], [1], [1, 1]]], 2),
    # 15. Non-square 3×1 column vector
    ("3x1_col_vector_f5", [[[0, 1]], [[2, 1]], [[4]]], 5),
    # 16. 2×4 wide matrix
    ("2x4_wide_f3", [[[1], [0, 1], [0, 0, 1], [1, 1]], [[0, 1], [1], [1, 0, 1], []]], 3),
    # 17. 4×2 tall matrix
    ("4x2_tall_f3", [[[1], [0, 1]], [[0, 1], [1]], [[1, 1], [0, 0, 1]], [[], [1]]], 3),
    # 18. 2×2 high degree: [[x^4+1, x^3], [x^2, x^4+x]] over F_2
    ("2x2_high_degree_f2", [[[1, 0, 0, 0, 1], [0, 0, 0, 1]], [[0, 0, 1], [0, 1, 0, 0, 1]]], 2),
    # 19. 3×3 companion matrix of x^3 + x + 1 over F_2:
    #    [[0, 0, 1], [1, 0, 1], [0, 1, 0]]
    ("3x3_companion_f2", [[[], [], [1]], [[1], [], [1]], [[], [1], []]], 2),
    # 20. 2×2 [[x^2+x+1, x^2+1], [x+1, x^2+x+1]] over F_2
    ("2x2_irreducible_f2", [[[1, 1, 1], [1, 0, 1]], [[1, 1], [1, 1, 1]]], 2),
    # 21. 3×3 with large prime F_97
    (
        "3x3_f97",
        [
            [[1, 50], [0, 0, 1], [3]],
            [[96, 1], [2, 3, 1], [0, 1]],
            [[10, 0, 1], [1], [50, 1]],
        ],
        97,
    ),
]


# ---------------------------------------------------------------------------
# SNF tests
# ---------------------------------------------------------------------------


class TestEvilPolySNF:
    @pytest.mark.parametrize("backend", _SNF_BACKENDS)
    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_snf_invariant_factors_monic(self, label, entries, p, backend):
        m = _dense(entries, p)
        result = poly_smith_normal_form(m, backend=backend)
        for f in result.invariant_factors:
            assert _is_monic_or_zero(f), (
                f"[{backend}] [{label}] factor {f} is not monic"
            )

    @pytest.mark.parametrize("backend", _SNF_BACKENDS)
    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_snf_diagonal_shape(self, label, entries, p, backend):
        m = _dense(entries, p)
        nrows = len(entries)
        ncols = len(entries[0]) if entries else 0
        result = poly_smith_normal_form(m, backend=backend)
        snf = result.smith_normal_form.entries
        assert len(snf) == nrows
        for row in snf:
            assert len(row) == ncols
        # Off-diagonal must be zero
        for i in range(nrows):
            for j in range(ncols):
                if i != j:
                    assert not snf[i][j], (
                        f"[{backend}] [{label}] snf[{i}][{j}]={snf[i][j]} not zero"
                    )

    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_snf_all_backends_agree(self, label, entries, p):
        if len(_SNF_BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-check")
        m = _dense(entries, p)
        ref = poly_smith_normal_form(m, backend=_SNF_BACKENDS[0])
        for b in _SNF_BACKENDS[1:]:
            other = poly_smith_normal_form(m, backend=b)
            assert ref.invariant_factors == other.invariant_factors, (
                f"[{label}] SNF differs: {_SNF_BACKENDS[0]}={ref.invariant_factors} "
                f"{b}={other.invariant_factors}"
            )


# ---------------------------------------------------------------------------
# SNF with transforms tests
# ---------------------------------------------------------------------------


class TestEvilPolySNFWithTransforms:
    @pytest.mark.parametrize("backend", _SNF_T_BACKENDS)
    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_snf_transform_equation(self, label, entries, p, backend):
        m = _dense(entries, p)
        result = poly_smith_normal_form_with_transforms(m, backend=backend)
        U = result.left_transform.entries
        V = result.right_transform.entries
        snf = result.smith_normal_form.entries
        UM = poly_mat_mul(U, entries, p)
        UMV = poly_mat_mul(UM, V, p)
        assert _mat_eq(UMV, snf), (
            f"[{backend}] [{label}] U @ M @ V != SNF\nUMV={UMV}\nSNF={snf}"
        )
        assert_invertible_poly(U, p, "U")
        assert_invertible_poly(V, p, "V")


# ---------------------------------------------------------------------------
# HNF tests
# ---------------------------------------------------------------------------


class TestEvilPolyHNF:
    @pytest.mark.parametrize("backend", _HNF_BACKENDS)
    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_hnf_upper_triangular(self, label, entries, p, backend):
        m = _dense(entries, p)
        result = poly_hermite_normal_form(m, backend=backend)
        H = result.hermite_normal_form.entries
        assert _is_upper_triangular(H), (
            f"[{backend}] [{label}] HNF not upper triangular: {H}"
        )

    @pytest.mark.parametrize("backend", _HNF_BACKENDS)
    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_hnf_monic_pivots(self, label, entries, p, backend):
        m = _dense(entries, p)
        result = poly_hermite_normal_form(m, backend=backend)
        H = result.hermite_normal_form.entries
        r = min(len(H), len(H[0]) if H else 0)
        for i in range(r):
            assert _is_monic_or_zero(H[i][i]), (
                f"[{backend}] [{label}] pivot H[{i}][{i}]={H[i][i]} not monic"
            )

    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_hnf_all_backends_agree(self, label, entries, p):
        if len(_HNF_BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-check")
        m = _dense(entries, p)
        ref = poly_hermite_normal_form(m, backend=_HNF_BACKENDS[0])
        for b in _HNF_BACKENDS[1:]:
            other = poly_hermite_normal_form(m, backend=b)
            assert _mat_eq(
                ref.hermite_normal_form.entries,
                other.hermite_normal_form.entries,
            ), f"[{label}] HNF differs: {_HNF_BACKENDS[0]} vs {b}"


# ---------------------------------------------------------------------------
# HNF with transform tests
# ---------------------------------------------------------------------------


class TestEvilPolyHNFWithTransform:
    @pytest.mark.parametrize("backend", _HNF_T_BACKENDS)
    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_hnf_transform_equation(self, label, entries, p, backend):
        m = _dense(entries, p)
        result = poly_hermite_normal_form_with_transform(m, backend=backend)
        U = result.left_transform.entries
        hnf = result.hermite_normal_form.entries
        UM = poly_mat_mul(U, entries, p)
        assert _mat_eq(UM, hnf), (
            f"[{backend}] [{label}] U @ M != HNF\nUM={UM}\nHNF={hnf}"
        )
        assert_invertible_poly(U, p, "U")


# ---------------------------------------------------------------------------
# Elementary divisors tests
# ---------------------------------------------------------------------------


class TestEvilPolyED:
    @pytest.mark.parametrize("backend", _ED_BACKENDS)
    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_ed_matches_snf(self, label, entries, p, backend):
        m = _dense(entries, p)
        ed = poly_elementary_divisors(m, backend=backend)
        snf = poly_smith_normal_form(m, backend=backend)
        assert ed.elementary_divisors == snf.invariant_factors, (
            f"[{backend}] [{label}] ED != SNF invariant factors"
        )

    @pytest.mark.parametrize("label,entries,p", _CASES, ids=[c[0] for c in _CASES])
    def test_ed_all_backends_agree(self, label, entries, p):
        if len(_ED_BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-check")
        m = _dense(entries, p)
        ref = poly_elementary_divisors(m, backend=_ED_BACKENDS[0])
        for b in _ED_BACKENDS[1:]:
            other = poly_elementary_divisors(m, backend=b)
            assert ref.elementary_divisors == other.elementary_divisors, (
                f"[{label}] ED differs: {_ED_BACKENDS[0]}={ref.elementary_divisors} "
                f"{b}={other.elementary_divisors}"
            )
