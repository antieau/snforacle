"""Tests for the MAGMA poly backend over F_p[x] (skipped if magma not on PATH)."""

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

pytestmark = pytest.mark.skipif(
    shutil.which("magma") is None,
    reason="magma binary not found on PATH",
)

BACKEND = "magma"


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


def _poly_eq(a: list[int], b: list[int]) -> bool:
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


_I2 = [[[1], []], [[], [1]]]
_Z2 = [[[], []], [[], []]]
_M2 = [[[0, 1], [1]], [[1], [0, 1]]]
_DIAG3 = [[[0, 1], [], []], [[], [0, 0, 1], []], [[], [], [0, 0, 0, 1]]]
_M23_p5 = [[[1, 1], [0, 0, 1], [2]], [[3], [1], [0, 1]]]
_M32_p5 = [[[0, 1], [2]], [[1], [3, 1]], [[0, 0, 1], []]]


# ---------------------------------------------------------------------------
# SNF tests
# ---------------------------------------------------------------------------


class TestMagmaPolySNF:
    def test_identity_2x2_f2(self):
        result = poly_smith_normal_form(_dense(_I2, 2), backend=BACKEND)
        assert result.invariant_factors == [[1], [1]]

    def test_zero_2x2_f2(self):
        result = poly_smith_normal_form(_dense(_Z2, 2), backend=BACKEND)
        assert result.invariant_factors == []

    def test_2x2_f2(self):
        result = poly_smith_normal_form(_dense(_M2, 2), backend=BACKEND)
        assert len(result.invariant_factors) == 2
        for f in result.invariant_factors:
            assert f and f[-1] == 1

    def test_diagonal_3x3_f3(self):
        result = poly_smith_normal_form(_dense(_DIAG3, 3), backend=BACKEND)
        pp = poly_smith_normal_form(_dense(_DIAG3, 3), backend="pure_python")
        assert result.invariant_factors == pp.invariant_factors

    def test_2x3_non_square(self):
        result = poly_smith_normal_form(_dense(_M23_p5, 5), backend=BACKEND)
        pp = poly_smith_normal_form(_dense(_M23_p5, 5), backend="pure_python")
        assert result.invariant_factors == pp.invariant_factors

    def test_3x2_non_square(self):
        result = poly_smith_normal_form(_dense(_M32_p5, 5), backend=BACKEND)
        pp = poly_smith_normal_form(_dense(_M32_p5, 5), backend="pure_python")
        assert result.invariant_factors == pp.invariant_factors


class TestMagmaPolySNFWithTransforms:
    def _verify(self, matrix_dict: dict, p: int) -> None:
        result = poly_smith_normal_form_with_transforms(matrix_dict, backend=BACKEND)
        U = result.left_transform.entries
        V = result.right_transform.entries
        snf = result.smith_normal_form.entries
        m_entries = matrix_dict["entries"]
        UM = poly_mat_mul(U, m_entries, p)
        UMV = poly_mat_mul(UM, V, p)
        assert _mat_eq(UMV, snf), f"U @ M @ V != SNF"
        assert_invertible_poly(U, p, "U")
        assert_invertible_poly(V, p, "V")

    def test_identity_2x2_f2(self):
        self._verify(_dense(_I2, 2), 2)

    def test_2x2_f2(self):
        self._verify(_dense(_M2, 2), 2)

    def test_diagonal_3x3_f3(self):
        self._verify(_dense(_DIAG3, 3), 3)

    def test_2x3_non_square(self):
        self._verify(_dense(_M23_p5, 5), 5)

    def test_3x2_non_square(self):
        self._verify(_dense(_M32_p5, 5), 5)


class TestMagmaPolyHNF:
    def test_identity_2x2_f2(self):
        result = poly_hermite_normal_form(_dense(_I2, 2), backend=BACKEND)
        assert _mat_eq(result.hermite_normal_form.entries, _I2)

    def test_zero_2x2_f2(self):
        result = poly_hermite_normal_form(_dense(_Z2, 2), backend=BACKEND)
        assert _mat_eq(result.hermite_normal_form.entries, _Z2)

    def test_matches_pure_python(self):
        for m_dict, p in [
            (_dense(_M2, 2), 2),
            (_dense(_DIAG3, 3), 3),
            (_dense(_M23_p5, 5), 5),
        ]:
            magma_r = poly_hermite_normal_form(m_dict, backend=BACKEND)
            pp_r = poly_hermite_normal_form(m_dict, backend="pure_python")
            assert _mat_eq(
                magma_r.hermite_normal_form.entries,
                pp_r.hermite_normal_form.entries,
            ), f"MAGMA HNF != pure_python HNF for p={p}"


class TestMagmaPolyHNFWithTransform:
    def _verify(self, matrix_dict: dict, p: int) -> None:
        result = poly_hermite_normal_form_with_transform(matrix_dict, backend=BACKEND)
        U = result.left_transform.entries
        hnf = result.hermite_normal_form.entries
        m_entries = matrix_dict["entries"]
        UM = poly_mat_mul(U, m_entries, p)
        assert _mat_eq(UM, hnf), f"U @ M != HNF"
        assert_invertible_poly(U, p, "U")

    def test_identity_2x2_f2(self):
        self._verify(_dense(_I2, 2), 2)

    def test_2x2_f2(self):
        self._verify(_dense(_M2, 2), 2)

    def test_diagonal_3x3_f3(self):
        self._verify(_dense(_DIAG3, 3), 3)

    def test_2x3_non_square(self):
        self._verify(_dense(_M23_p5, 5), 5)

    def test_3x2_non_square(self):
        self._verify(_dense(_M32_p5, 5), 5)


class TestMagmaPolyED:
    def test_matches_snf(self):
        for m_dict in [_dense(_I2, 2), _dense(_M2, 2), _dense(_DIAG3, 3), _dense(_M23_p5, 5)]:
            ed = poly_elementary_divisors(m_dict, backend=BACKEND)
            snf = poly_smith_normal_form(m_dict, backend=BACKEND)
            assert ed.elementary_divisors == snf.invariant_factors

    def test_matches_pure_python(self):
        for m_dict, p in [(_dense(_M2, 2), 2), (_dense(_DIAG3, 3), 3)]:
            magma_ed = poly_elementary_divisors(m_dict, backend=BACKEND)
            pp_ed = poly_elementary_divisors(m_dict, backend="pure_python")
            assert magma_ed.elementary_divisors == pp_ed.elementary_divisors


# ---------------------------------------------------------------------------
# Cross-backend consistency: magma vs pure_python (parametrized)
# ---------------------------------------------------------------------------


_CROSS_CASES = [
    (_dense(_I2, 2), 2, "identity_2x2_f2"),
    (_dense(_Z2, 2), 2, "zero_2x2_f2"),
    (_dense(_M2, 2), 2, "M2_f2"),
    (_dense(_DIAG3, 3), 3, "diag_3x3_f3"),
    (_dense(_M23_p5, 5), 5, "M23_f5"),
    (_dense(_M32_p5, 5), 5, "M32_f5"),
]


@pytest.mark.parametrize("m_dict,p,name", _CROSS_CASES, ids=[c[2] for c in _CROSS_CASES])
def test_magma_vs_pure_python_snf(m_dict, p, name):
    magma_r = poly_smith_normal_form(m_dict, backend="magma")
    pp_r = poly_smith_normal_form(m_dict, backend="pure_python")
    assert magma_r.invariant_factors == pp_r.invariant_factors, (
        f"SNF differs ({name}): magma={magma_r.invariant_factors} pp={pp_r.invariant_factors}"
    )


@pytest.mark.parametrize("m_dict,p,name", _CROSS_CASES, ids=[c[2] for c in _CROSS_CASES])
def test_magma_vs_pure_python_hnf(m_dict, p, name):
    magma_r = poly_hermite_normal_form(m_dict, backend="magma")
    pp_r = poly_hermite_normal_form(m_dict, backend="pure_python")
    assert _mat_eq(
        magma_r.hermite_normal_form.entries,
        pp_r.hermite_normal_form.entries,
    ), f"HNF differs ({name})"
