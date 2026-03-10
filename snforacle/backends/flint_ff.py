"""FLINT backend for matrix operations over F_p via python-flint's nmod_mat.

For operations without transforms, ``nmod_mat.rref()`` is used directly —
this is highly optimised (SIMD, cache-friendly) and significantly faster than
pure Python for large matrices.

For operations that require explicit transform matrices (SNF+T, HNF+T) the
pure-Python modular Gaussian elimination from ``pure_python_ff`` is used,
since python-flint 0.8.0 does not expose transform tracking from ``rref()``.
"""

from __future__ import annotations

from snforacle.backends.ff_base import FFBackend
from snforacle.backends.pure_python_ff import _hnf_with_transform, _snf_with_transforms


def _flint():
    try:
        import flint
    except ImportError as exc:
        raise ImportError(
            "python-flint is required for the 'flint' FF backend. "
            "Install it with: pip install snforacle[flint]"
        ) from exc
    return flint


class FlintFFBackend(FFBackend):
    """Uses python-flint's ``nmod_mat`` for fast F_p matrix operations.

    Notes
    -----
    ``nmod_mat.rref()`` computes the reduced row echelon form in-place and
    returns the rank.  This is used for all no-transform operations.  For
    operations requiring explicit unimodular matrices (SNF+T, HNF+T) the
    pure-Python algorithm is used instead, since python-flint 0.8.0 does
    not expose transform tracking from ``nmod_mat``.
    """

    def __init__(self) -> None:
        self._flint = _flint()  # fail fast if python-flint is not installed

    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int]:
        mat = self._flint.nmod_mat(matrix, p)
        _, rank = mat.rref()
        # SNF = diag(1,...,1,0,...,0) — just need the rank.
        snf = [[0] * ncols for _ in range(nrows)]
        for i in range(rank):
            snf[i][i] = 1
        return snf, rank

    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int, list[list[int]], list[list[int]]]:
        return _snf_with_transforms(matrix, nrows, ncols, p)

    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]]]:
        mat = self._flint.nmod_mat(matrix, p)
        rref_mat, _ = mat.rref()
        hnf = [[int(rref_mat[i, j]) for j in range(ncols)] for i in range(nrows)]
        return (hnf,)

    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        return _hnf_with_transform(matrix, nrows, ncols, p)

    def compute_rank(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> int:
        mat = self._flint.nmod_mat(matrix, p)
        _, rank = mat.rref()
        return rank
