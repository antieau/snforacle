"""Public API for snforacle matrix operations over finite fields F_p.

Entry points:

``ff_smith_normal_form(matrix, backend)``
    Returns the SNF (diag(1,...,1,0,...,0)) and rank.

``ff_smith_normal_form_with_transforms(matrix, backend)``
    Returns the SNF together with invertible left and right transforms U, V
    such that U @ M @ V = SNF.

``ff_hermite_normal_form(matrix, backend)``
    Returns the row Hermite Normal Form (= RREF over a field).

``ff_hermite_normal_form_with_transform(matrix, backend)``
    Returns HNF together with the left invertible transform U.

``ff_rank(matrix, backend)``
    Returns the rank of the matrix over F_p.

All functions accept *matrix* as either a validated ``DenseFFMatrix`` /
``SparseFFMatrix`` Pydantic model, or a plain ``dict`` that will be
validated against those schemas.
"""

from __future__ import annotations

import shutil as _shutil
from typing import Any, Literal

from pydantic import TypeAdapter

from snforacle.backends.ff_base import FFBackend
from snforacle.ff_schema import (
    DenseFFMatrix,
    DenseFFMatrixOut,
    FFHNFResult,
    FFHNFWithTransformResult,
    FFMatrix,
    FFRankResult,
    FFSNFResult,
    FFSNFWithTransformsResult,
    SparseFFMatrix,
)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[FFBackend]] = {}


def _register(name: str, cls: type[FFBackend]) -> None:
    _BACKENDS[name] = cls


def _get_backend(name: str) -> FFBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown FF backend {name!r}. Available backends: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]()


def _lazy_register_pure_python_ff() -> None:
    from snforacle.backends.pure_python_ff import PurePythonFFBackend
    _register("pure_python", PurePythonFFBackend)


def _lazy_register_flint_ff() -> None:
    from snforacle.backends.flint_ff import FlintFFBackend
    _register("flint", FlintFFBackend)


def _lazy_register_sage_ff() -> None:
    from snforacle.backends.sage_ff import SageFFBackend
    _register("sage", SageFFBackend)


def _lazy_register_magma_ff() -> None:
    from snforacle.backends.magma_ff import MagmaFFBackend
    _register("magma", MagmaFFBackend)


_lazy_register_pure_python_ff()
_lazy_register_flint_ff()
_lazy_register_sage_ff()
_lazy_register_magma_ff()

# ---------------------------------------------------------------------------
# Default backend detection
# ---------------------------------------------------------------------------


def _detect_available_ff_backends() -> dict[str, bool]:
    avail: dict[str, bool] = {"pure_python": True}
    try:
        import flint  # noqa: F401
        avail["flint"] = True
    except ImportError:
        avail["flint"] = False
    avail["sage"] = _shutil.which("sage") is not None
    avail["magma"] = _shutil.which("magma") is not None
    return avail


_OP_PRIORITY: dict[str, list[str]] = {
    "snf":            ["flint", "pure_python", "sage", "magma"],
    "snf_transforms": ["pure_python", "sage", "magma"],
    "hnf":            ["flint", "pure_python", "sage", "magma"],
    "hnf_transform":  ["pure_python", "sage", "magma"],
    "rank":           ["flint", "pure_python", "sage", "magma"],
}

_available = _detect_available_ff_backends()


def _best_backend(op: str) -> str:
    for b in _OP_PRIORITY[op]:
        if _available.get(b):
            return b
    return "pure_python"


_DEFAULTS: dict[str, str] = {op: _best_backend(op) for op in _OP_PRIORITY}

# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

_matrix_adapter: TypeAdapter[DenseFFMatrix | SparseFFMatrix] = TypeAdapter(FFMatrix)  # type: ignore[type-arg]


def _parse_matrix(matrix: Any) -> DenseFFMatrix | SparseFFMatrix:
    if isinstance(matrix, (DenseFFMatrix, SparseFFMatrix)):
        return matrix
    return _matrix_adapter.validate_python(matrix)


def _to_ff_out(
    entries: list[list[int]], nrows: int, ncols: int, p: int
) -> DenseFFMatrixOut:
    return DenseFFMatrixOut(nrows=nrows, ncols=ncols, p=p, entries=entries)


def _identity_ff(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


_FFBackend = Literal["flint", "pure_python", "sage", "magma"]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ff_smith_normal_form(
    matrix: Any,
    backend: _FFBackend | None = None,
) -> FFSNFResult:
    """Compute the Smith normal form of a matrix over F_p.

    Over a finite field every nonzero element is a unit, so the SNF is
    always ``diag(1, ..., 1, 0, ..., 0)`` where the number of 1s equals
    the rank.

    Parameters
    ----------
    matrix:
        The input matrix as a ``DenseFFMatrix``, ``SparseFFMatrix``, or a
        plain ``dict`` conforming to one of those schemas.
    backend:
        Name of the backend to use: ``"flint"``, ``"pure_python"``,
        ``"sage"``, or ``"magma"``.  Defaults to ``"flint"`` if available,
        then ``"pure_python"``.

    Returns
    -------
    FFSNFResult
        A Pydantic model with fields:

        ``smith_normal_form``
            The m×n SNF matrix over F_p with 1s at (0,0),...,(r-1,r-1).
        ``rank``
            The rank r of the matrix over F_p.
    """
    if backend is None:
        backend = _DEFAULTS["snf"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        snf = [[0] * m.ncols for _ in range(m.nrows)]
        return FFSNFResult(
            smith_normal_form=_to_ff_out(snf, m.nrows, m.ncols, m.p),
            rank=0,
        )
    dense = m.to_dense()
    snf_mat, rank = _get_backend(backend).compute_snf(dense, m.nrows, m.ncols, m.p)
    return FFSNFResult(
        smith_normal_form=_to_ff_out(snf_mat, m.nrows, m.ncols, m.p),
        rank=rank,
    )


def ff_smith_normal_form_with_transforms(
    matrix: Any,
    backend: _FFBackend | None = None,
) -> FFSNFWithTransformsResult:
    """Compute the SNF together with invertible left and right transforms over F_p.

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`ff_smith_normal_form`).
    backend:
        Name of the backend to use. Defaults to ``"pure_python"``.

    Returns
    -------
    FFSNFWithTransformsResult
        A Pydantic model with fields:

        ``smith_normal_form``
            The m×n SNF matrix over F_p.
        ``rank``
            The rank r of the matrix over F_p.
        ``left_transform``
            Invertible m×m matrix U over F_p.
        ``right_transform``
            Invertible n×n matrix V over F_p.

        Satisfies: ``left_transform @ matrix @ right_transform = smith_normal_form``.
    """
    if backend is None:
        backend = _DEFAULTS["snf_transforms"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        snf = [[0] * m.ncols for _ in range(m.nrows)]
        return FFSNFWithTransformsResult(
            smith_normal_form=_to_ff_out(snf, m.nrows, m.ncols, m.p),
            rank=0,
            left_transform=_to_ff_out(_identity_ff(m.nrows), m.nrows, m.nrows, m.p),
            right_transform=_to_ff_out(_identity_ff(m.ncols), m.ncols, m.ncols, m.p),
        )
    dense = m.to_dense()
    snf_mat, rank, left, right = _get_backend(backend).compute_snf_with_transforms(
        dense, m.nrows, m.ncols, m.p
    )
    return FFSNFWithTransformsResult(
        smith_normal_form=_to_ff_out(snf_mat, m.nrows, m.ncols, m.p),
        rank=rank,
        left_transform=_to_ff_out(left, m.nrows, m.nrows, m.p),
        right_transform=_to_ff_out(right, m.ncols, m.ncols, m.p),
    )


def ff_hermite_normal_form(
    matrix: Any,
    backend: _FFBackend | None = None,
) -> FFHNFResult:
    """Compute the row Hermite Normal Form of a matrix over F_p.

    Over a finite field the HNF is the unique reduced row echelon form (RREF):
    upper-staircase with leading 1 in each nonzero row, and all other entries
    in each pivot column equal to 0.

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`ff_smith_normal_form`).
    backend:
        Name of the backend to use. Defaults to ``"flint"`` if available,
        then ``"pure_python"``.

    Returns
    -------
    FFHNFResult
        A Pydantic model with field:

        ``hermite_normal_form``
            The unique RREF of the matrix over F_p.
    """
    if backend is None:
        backend = _DEFAULTS["hnf"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        hnf = [[0] * m.ncols for _ in range(m.nrows)]
        return FFHNFResult(
            hermite_normal_form=_to_ff_out(hnf, m.nrows, m.ncols, m.p)
        )
    dense = m.to_dense()
    (hnf_mat,) = _get_backend(backend).compute_hnf(dense, m.nrows, m.ncols, m.p)
    return FFHNFResult(
        hermite_normal_form=_to_ff_out(hnf_mat, m.nrows, m.ncols, m.p),
    )


def ff_hermite_normal_form_with_transform(
    matrix: Any,
    backend: _FFBackend | None = None,
) -> FFHNFWithTransformResult:
    """Compute the row HNF together with the left invertible transform over F_p.

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`ff_smith_normal_form`).
    backend:
        Name of the backend to use. Defaults to ``"pure_python"``.

    Returns
    -------
    FFHNFWithTransformResult
        A Pydantic model with fields:

        ``hermite_normal_form``
            The unique RREF of the matrix over F_p.
        ``left_transform``
            Invertible m×m matrix U over F_p.

        Satisfies: ``left_transform @ matrix = hermite_normal_form``.
    """
    if backend is None:
        backend = _DEFAULTS["hnf_transform"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        hnf = [[0] * m.ncols for _ in range(m.nrows)]
        return FFHNFWithTransformResult(
            hermite_normal_form=_to_ff_out(hnf, m.nrows, m.ncols, m.p),
            left_transform=_to_ff_out(_identity_ff(m.nrows), m.nrows, m.nrows, m.p),
        )
    dense = m.to_dense()
    hnf_mat, left = _get_backend(backend).compute_hnf_with_transform(
        dense, m.nrows, m.ncols, m.p
    )
    return FFHNFWithTransformResult(
        hermite_normal_form=_to_ff_out(hnf_mat, m.nrows, m.ncols, m.p),
        left_transform=_to_ff_out(left, m.nrows, m.nrows, m.p),
    )


def ff_rank(
    matrix: Any,
    backend: _FFBackend | None = None,
) -> FFRankResult:
    """Compute the rank of a matrix over F_p.

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`ff_smith_normal_form`).
    backend:
        Name of the backend to use. Defaults to ``"flint"`` if available,
        then ``"pure_python"``.

    Returns
    -------
    FFRankResult
        A Pydantic model with field:

        ``rank``
            The rank of the matrix over F_p.
    """
    if backend is None:
        backend = _DEFAULTS["rank"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        return FFRankResult(rank=0)
    dense = m.to_dense()
    rank = _get_backend(backend).compute_rank(dense, m.nrows, m.ncols, m.p)
    return FFRankResult(rank=rank)
