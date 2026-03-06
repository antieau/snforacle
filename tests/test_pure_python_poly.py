"""Tests for the pure_python poly backend and its operations over F_p[x]."""

from __future__ import annotations

import pytest

from snforacle import (
    poly_elementary_divisors,
    poly_hermite_normal_form,
    poly_hermite_normal_form_with_transform,
    poly_smith_normal_form,
    poly_smith_normal_form_with_transforms,
)
from snforacle.backends.pure_python_poly import poly_mat_mul
from snforacle.poly_schema import PolyHNFResult, PolySNFResult

BACKEND = "pure_python"


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


def _sparse(nrows: int, ncols: int, p: int, entries: list[tuple[int, int, list[int]]]) -> dict:
    return {
        "format": "sparse_poly",
        "nrows": nrows,
        "ncols": ncols,
        "p": p,
        "entries": [{"row": r, "col": c, "coeffs": poly} for r, c, poly in entries],
    }


def _poly_eq(a: list[int], b: list[int]) -> bool:
    """Trim and compare two polynomials."""
    def trim(x):
        c = list(x)
        while c and c[-1] == 0:
            c.pop()
        return c
    return trim(a) == trim(b)


def _mat_eq(A: list[list[list[int]]], B: list[list[list[int]]]) -> bool:
    if len(A) != len(B):
        return False
    for ra, rb in zip(A, B):
        if len(ra) != len(rb):
            return False
        for pa, pb in zip(ra, rb):
            if not _poly_eq(pa, pb):
                return False
    return True


# ---------------------------------------------------------------------------
# Basic polynomial matrix helpers
# ---------------------------------------------------------------------------

# Identity 2x2 over F_2[x]: [[1, 0], [0, 1]]
_I2 = [[[1], []], [[], [1]]]

# Zero 2x2 over F_2[x]
_Z2 = [[[], []], [[], []]]

# Simple 2x2 over F_2[x]: [[x, 1], [1, x]]
_M2 = [[[0, 1], [1]], [[1], [0, 1]]]

# 3x3 over F_3[x]: diagonal with x, x^2, x^3
_DIAG3 = [
    [[0, 1], [], []],
    [[], [0, 0, 1], []],
    [[], [], [0, 0, 0, 1]],
]

# 2x3 over F_5[x]: [[x+1, x^2, 2], [3, 1, x]]
_M23_p5 = [[[1, 1], [0, 0, 1], [2]], [[3], [1], [0, 1]]]

# 3x2 over F_5[x]: [[x, 2], [1, x+3], [x^2, 0]]
_M32_p5 = [[[0, 1], [2]], [[1], [3, 1]], [[0, 0, 1], []]]


# ---------------------------------------------------------------------------
# SNF tests
# ---------------------------------------------------------------------------


class TestPurePythonPolySNF:
    """Test SNF over F_p[x] via the pure_python backend."""

    def test_identity_2x2_f2(self):
        result = poly_smith_normal_form(_dense(_I2, 2), backend=BACKEND)
        assert result.invariant_factors == [[1], [1]]

    def test_zero_2x2_f2(self):
        result = poly_smith_normal_form(_dense(_Z2, 2), backend=BACKEND)
        assert result.invariant_factors == []
        assert _mat_eq(result.smith_normal_form.entries, _Z2)

    def test_diagonal_3x3_f3(self):
        # diag(x, x^2, x^3) → SNF should be diag(x, x^2, x^3) since x | x^2 | x^3
        result = poly_smith_normal_form(_dense(_DIAG3, 3), backend=BACKEND)
        assert len(result.invariant_factors) == 3
        # All factors should be monic
        for f in result.invariant_factors:
            assert f and f[-1] == 1

    def test_2x2_f2(self):
        # [[x, 1], [1, x]] over F_2: det = x^2 + 1 = (x+1)^2
        result = poly_smith_normal_form(_dense(_M2, 2), backend=BACKEND)
        assert len(result.invariant_factors) == 2
        # Both factors should be monic
        for f in result.invariant_factors:
            assert f and f[-1] == 1

    def test_2x3_non_square(self):
        result = poly_smith_normal_form(_dense(_M23_p5, 5), backend=BACKEND)
        snf = result.smith_normal_form.entries
        assert len(snf) == 2 and len(snf[0]) == 3
        # Off-diagonal should be zero
        for i in range(2):
            for j in range(3):
                if i != j:
                    assert not snf[i][j], f"snf[{i}][{j}] = {snf[i][j]} should be zero"

    def test_3x2_non_square(self):
        result = poly_smith_normal_form(_dense(_M32_p5, 5), backend=BACKEND)
        snf = result.smith_normal_form.entries
        assert len(snf) == 3 and len(snf[0]) == 2

    def test_return_type(self):
        result = poly_smith_normal_form(_dense(_I2, 2), backend=BACKEND)
        assert isinstance(result, PolySNFResult)

    def test_sparse_input(self):
        # sparse: 2x2 identity over F_7
        sparse = _sparse(2, 2, 7, [(0, 0, [1]), (1, 1, [1])])
        result = poly_smith_normal_form(sparse, backend=BACKEND)
        assert result.invariant_factors == [[1], [1]]

    def test_scalar_matrix_over_fp(self):
        # [[3]] over F_7 → SNF = [[1]] (since 3 is a unit in F_7)
        result = poly_smith_normal_form(_dense([[[3]]], 7), backend=BACKEND)
        assert result.invariant_factors == [[1]]

    def test_zero_matrix_0x0(self):
        inp = {"format": "dense_poly", "nrows": 0, "ncols": 0, "p": 5, "entries": []}
        result = poly_smith_normal_form(inp, backend=BACKEND)
        assert result.invariant_factors == []

    def test_invariant_factors_monic(self):
        # All invariant factors must be monic (leading coeff == 1)
        result = poly_smith_normal_form(_dense(_M2, 2), backend=BACKEND)
        for f in result.invariant_factors:
            assert f[-1] == 1, f"Factor {f} is not monic"

    def test_divisibility_chain(self):
        # Invariant factors must satisfy d_i | d_{i+1}
        from snforacle.backends.pure_python_poly import _divmod_poly
        result = poly_smith_normal_form(_dense(_DIAG3, 3), backend=BACKEND)
        factors = result.invariant_factors
        p = 3
        for i in range(len(factors) - 1):
            _, r = _divmod_poly(factors[i + 1], factors[i], p)
            assert not r, f"d_{i} does not divide d_{i+1}: {factors[i]} ∤ {factors[i+1]}"


# ---------------------------------------------------------------------------
# SNF with transforms tests
# ---------------------------------------------------------------------------


class TestPurePythonPolySNFWithTransforms:
    """Test that U @ M @ V == SNF over F_p[x]."""

    def _verify(self, matrix_dict: dict, p: int) -> None:
        result = poly_smith_normal_form_with_transforms(matrix_dict, backend=BACKEND)
        U = result.left_transform.entries
        V = result.right_transform.entries
        snf = result.smith_normal_form.entries
        # Reconstruct original from dict
        orig = matrix_dict
        m_entries = orig["entries"]
        # Compute U @ M @ V
        UM = poly_mat_mul(U, m_entries, p)
        UMV = poly_mat_mul(UM, V, p)
        assert _mat_eq(UMV, snf), f"U @ M @ V != SNF\nUMV={UMV}\nSNF={snf}"

    def test_identity_2x2_f2(self):
        self._verify(_dense(_I2, 2), 2)

    def test_zero_2x2_f2(self):
        self._verify(_dense(_Z2, 2), 2)

    def test_2x2_f2(self):
        self._verify(_dense(_M2, 2), 2)

    def test_diagonal_3x3_f3(self):
        self._verify(_dense(_DIAG3, 3), 3)

    def test_2x3_non_square(self):
        self._verify(_dense(_M23_p5, 5), 5)

    def test_3x2_non_square(self):
        self._verify(_dense(_M32_p5, 5), 5)

    def test_0x0_empty(self):
        inp = {"format": "dense_poly", "nrows": 0, "ncols": 0, "p": 5, "entries": []}
        result = poly_smith_normal_form_with_transforms(inp, backend=BACKEND)
        assert result.invariant_factors == []


# ---------------------------------------------------------------------------
# HNF tests
# ---------------------------------------------------------------------------


class TestPurePythonPolyHNF:
    """Test row HNF over F_p[x] via the pure_python backend."""

    def test_identity_2x2_f2(self):
        result = poly_hermite_normal_form(_dense(_I2, 2), backend=BACKEND)
        assert _mat_eq(result.hermite_normal_form.entries, _I2)

    def test_zero_2x2_f2(self):
        result = poly_hermite_normal_form(_dense(_Z2, 2), backend=BACKEND)
        assert _mat_eq(result.hermite_normal_form.entries, _Z2)

    def test_upper_triangular(self):
        # HNF must be upper triangular
        result = poly_hermite_normal_form(_dense(_M2, 2), backend=BACKEND)
        H = result.hermite_normal_form.entries
        for i in range(len(H)):
            for j in range(i):
                assert not H[i][j], f"H[{i}][{j}] = {H[i][j]} is below diagonal"

    def test_monic_pivots(self):
        # All nonzero pivot entries (diagonal) must be monic
        result = poly_hermite_normal_form(_dense(_M2, 2), backend=BACKEND)
        H = result.hermite_normal_form.entries
        for i in range(min(len(H), len(H[0]) if H else 0)):
            if H[i][i]:
                assert H[i][i][-1] == 1, f"Pivot H[{i}][{i}] = {H[i][i]} not monic"

    def test_2x3_non_square(self):
        result = poly_hermite_normal_form(_dense(_M23_p5, 5), backend=BACKEND)
        H = result.hermite_normal_form.entries
        assert len(H) == 2 and len(H[0]) == 3

    def test_3x2_non_square(self):
        result = poly_hermite_normal_form(_dense(_M32_p5, 5), backend=BACKEND)
        H = result.hermite_normal_form.entries
        assert len(H) == 3 and len(H[0]) == 2

    def test_return_type(self):
        result = poly_hermite_normal_form(_dense(_I2, 2), backend=BACKEND)
        assert isinstance(result, PolyHNFResult)


# ---------------------------------------------------------------------------
# HNF with transform tests
# ---------------------------------------------------------------------------


class TestPurePythonPolyHNFWithTransform:
    """Test that U @ M == HNF over F_p[x]."""

    def _verify(self, matrix_dict: dict, p: int) -> None:
        result = poly_hermite_normal_form_with_transform(matrix_dict, backend=BACKEND)
        U = result.left_transform.entries
        hnf = result.hermite_normal_form.entries
        m_entries = matrix_dict["entries"]
        UM = poly_mat_mul(U, m_entries, p)
        assert _mat_eq(UM, hnf), f"U @ M != HNF\nUM={UM}\nHNF={hnf}"

    def test_identity_2x2_f2(self):
        self._verify(_dense(_I2, 2), 2)

    def test_zero_2x2_f2(self):
        self._verify(_dense(_Z2, 2), 2)

    def test_2x2_f2(self):
        self._verify(_dense(_M2, 2), 2)

    def test_diagonal_3x3_f3(self):
        self._verify(_dense(_DIAG3, 3), 3)

    def test_2x3_non_square(self):
        self._verify(_dense(_M23_p5, 5), 5)

    def test_3x2_non_square(self):
        self._verify(_dense(_M32_p5, 5), 5)


# ---------------------------------------------------------------------------
# Elementary divisors tests
# ---------------------------------------------------------------------------


class TestPurePythonPolyED:
    """Test elementary divisors over F_p[x]."""

    def test_matches_snf(self):
        for m_dict, p in [
            (_dense(_I2, 2), 2),
            (_dense(_M2, 2), 2),
            (_dense(_DIAG3, 3), 3),
            (_dense(_M23_p5, 5), 5),
        ]:
            ed = poly_elementary_divisors(m_dict, backend=BACKEND)
            snf = poly_smith_normal_form(m_dict, backend=BACKEND)
            assert ed.elementary_divisors == snf.invariant_factors, (
                f"ED != SNF invariant factors for p={p}"
            )

    def test_zero_matrix(self):
        result = poly_elementary_divisors(_dense(_Z2, 2), backend=BACKEND)
        assert result.elementary_divisors == []

    def test_identity(self):
        result = poly_elementary_divisors(_dense(_I2, 2), backend=BACKEND)
        assert result.elementary_divisors == [[1], [1]]
