"""Public API for snforacle polynomial matrix operations over F_p[x].

Entry points:

``poly_smith_normal_form(matrix, backend)``
    Returns the SNF diagonal matrix and monic invariant factors.

``poly_smith_normal_form_with_transforms(matrix, backend)``
    Returns the SNF together with invertible left and right transformation
    matrices U, V such that U @ M @ V = SNF.

``poly_hermite_normal_form(matrix, backend)``
    Returns the row Hermite Normal Form H = U @ M.

``poly_hermite_normal_form_with_transform(matrix, backend)``
    Returns HNF together with the left invertible transform U.

``poly_elementary_divisors(matrix, backend)``
    Returns the non-zero invariant factors (same as SNF diagonal).

All functions accept *matrix* as either a validated ``DensePolyMatrix`` /
``SparsePolyMatrix`` Pydantic model, or a plain ``dict`` that will be
validated against those schemas.
"""

from __future__ import annotations

import shutil as _shutil
from typing import Any

from pydantic import TypeAdapter

from snforacle.backends.poly_base import PolyBackend
from snforacle.poly_schema import (
    DensePolyMatrix,
    DensePolyMatrixOut,
    PolyElementaryDivisorsResult,
    PolyHNFResult,
    PolyHNFWithTransformResult,
    PolyMatrix,
    PolySNFResult,
    PolySNFWithTransformsResult,
    SparsePolyMatrix,
)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[PolyBackend]] = {}


def _register(name: str, cls: type[PolyBackend]) -> None:
    _BACKENDS[name] = cls


def _get_backend(name: str) -> PolyBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown poly backend {name!r}. Available backends: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]()


def _lazy_register_pure_python_poly() -> None:
    from snforacle.backends.pure_python_poly import PurePythonPolyBackend
    _register("pure_python", PurePythonPolyBackend)


def _lazy_register_sage_poly() -> None:
    from snforacle.backends.sage_poly import SagePolyBackend
    _register("sage", SagePolyBackend)


def _lazy_register_magma_poly() -> None:
    from snforacle.backends.magma_poly import MagmaPolyBackend
    _register("magma", MagmaPolyBackend)


_lazy_register_pure_python_poly()
_lazy_register_sage_poly()
_lazy_register_magma_poly()

# ---------------------------------------------------------------------------
# Default backend detection
# ---------------------------------------------------------------------------


def _detect_available_poly_backends() -> dict[str, bool]:
    avail: dict[str, bool] = {"pure_python": True}
    avail["sage"] = _shutil.which("sage") is not None
    avail["magma"] = _shutil.which("magma") is not None
    return avail


_OP_PRIORITY: dict[str, list[str]] = {
    "snf":            ["sage", "magma", "pure_python"],
    "snf_transforms": ["sage", "magma", "pure_python"],
    "hnf":            ["sage", "magma", "pure_python"],
    "hnf_transform":  ["sage", "magma", "pure_python"],
    "ed":             ["sage", "magma", "pure_python"],
}

_available = _detect_available_poly_backends()


def _best_backend(op: str) -> str:
    for b in _OP_PRIORITY[op]:
        if _available.get(b):
            return b
    return "pure_python"


_DEFAULTS: dict[str, str] = {op: _best_backend(op) for op in _OP_PRIORITY}

# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

_matrix_adapter: TypeAdapter[DensePolyMatrix | SparsePolyMatrix] = TypeAdapter(PolyMatrix)  # type: ignore[type-arg]


def _parse_matrix(matrix: Any) -> DensePolyMatrix | SparsePolyMatrix:
    if isinstance(matrix, (DensePolyMatrix, SparsePolyMatrix)):
        return matrix
    return _matrix_adapter.validate_python(matrix)


def _to_dense_poly_out(
    entries: list[list[list[int]]], nrows: int, ncols: int, p: int
) -> DensePolyMatrixOut:
    return DensePolyMatrixOut(nrows=nrows, ncols=ncols, p=p, entries=entries)


def _zero_poly_matrix(nrows: int, ncols: int) -> list[list[list[int]]]:
    return [[[] for _ in range(ncols)] for _ in range(nrows)]


def _identity_poly_matrix(n: int) -> list[list[list[int]]]:
    return [[[1] if i == j else [] for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def poly_smith_normal_form(
    matrix: Any,
    backend: str | None = None,
) -> PolySNFResult:
    """Compute the Smith normal form of a polynomial matrix over F_p[x].

    Parameters
    ----------
    matrix:
        The input matrix as a ``DensePolyMatrix``, ``SparsePolyMatrix``, or a
        plain ``dict`` conforming to one of those schemas.
    backend:
        Name of the backend to use. Defaults to ``"sage"`` if available,
        then ``"magma"``, then ``"pure_python"``.

    Returns
    -------
    PolySNFResult
        A Pydantic model with fields:

        ``smith_normal_form``
            The m×n SNF matrix. The diagonal entries d_1 | d_2 | ... | d_r
            are monic polynomials in ascending-degree order; all other
            entries are zero.
        ``invariant_factors``
            The sequence ``[d_1, ..., d_r]`` of nonzero monic diagonal entries.
    """
    if backend is None:
        backend = _DEFAULTS["snf"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        return PolySNFResult(
            smith_normal_form=_to_dense_poly_out(
                _zero_poly_matrix(m.nrows, m.ncols), m.nrows, m.ncols, m.p
            ),
            invariant_factors=[],
        )
    dense = m.to_dense()
    snf_mat, inv = _get_backend(backend).compute_snf(dense, m.nrows, m.ncols, m.p)
    return PolySNFResult(
        smith_normal_form=_to_dense_poly_out(snf_mat, m.nrows, m.ncols, m.p),
        invariant_factors=inv,
    )


def poly_smith_normal_form_with_transforms(
    matrix: Any,
    backend: str | None = None,
) -> PolySNFWithTransformsResult:
    """Compute the SNF together with invertible left and right transforms.

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`poly_smith_normal_form`).
    backend:
        Name of the backend to use.

    Returns
    -------
    PolySNFWithTransformsResult
        A Pydantic model with fields:

        ``smith_normal_form``
            The m×n SNF matrix.
        ``invariant_factors``
            The nonzero monic diagonal entries.
        ``left_transform``
            Invertible m×m matrix U over F_p[x].
        ``right_transform``
            Invertible n×n matrix V over F_p[x].

        Satisfies: ``left_transform @ matrix @ right_transform = smith_normal_form``.
    """
    if backend is None:
        backend = _DEFAULTS["snf_transforms"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        return PolySNFWithTransformsResult(
            smith_normal_form=_to_dense_poly_out(
                _zero_poly_matrix(m.nrows, m.ncols), m.nrows, m.ncols, m.p
            ),
            invariant_factors=[],
            left_transform=_to_dense_poly_out(
                _identity_poly_matrix(m.nrows), m.nrows, m.nrows, m.p
            ),
            right_transform=_to_dense_poly_out(
                _identity_poly_matrix(m.ncols), m.ncols, m.ncols, m.p
            ),
        )
    dense = m.to_dense()
    snf_mat, inv, left, right = _get_backend(backend).compute_snf_with_transforms(
        dense, m.nrows, m.ncols, m.p
    )
    return PolySNFWithTransformsResult(
        smith_normal_form=_to_dense_poly_out(snf_mat, m.nrows, m.ncols, m.p),
        invariant_factors=inv,
        left_transform=_to_dense_poly_out(left, m.nrows, m.nrows, m.p),
        right_transform=_to_dense_poly_out(right, m.ncols, m.ncols, m.p),
    )


def poly_hermite_normal_form(
    matrix: Any,
    backend: str | None = None,
) -> PolyHNFResult:
    """Compute the row Hermite Normal Form of a polynomial matrix over F_p[x].

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`poly_smith_normal_form`).
    backend:
        Name of the backend to use.

    Returns
    -------
    PolyHNFResult
        A Pydantic model with field:

        ``hermite_normal_form``
            Upper-triangular matrix with monic pivots; entries above each
            pivot have strictly smaller degree than the pivot.
    """
    if backend is None:
        backend = _DEFAULTS["hnf"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        return PolyHNFResult(
            hermite_normal_form=_to_dense_poly_out(
                _zero_poly_matrix(m.nrows, m.ncols), m.nrows, m.ncols, m.p
            )
        )
    dense = m.to_dense()
    (hnf,) = _get_backend(backend).compute_hnf(dense, m.nrows, m.ncols, m.p)
    return PolyHNFResult(
        hermite_normal_form=_to_dense_poly_out(hnf, m.nrows, m.ncols, m.p),
    )


def poly_hermite_normal_form_with_transform(
    matrix: Any,
    backend: str | None = None,
) -> PolyHNFWithTransformResult:
    """Compute the row HNF together with the left invertible transform.

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`poly_smith_normal_form`).
    backend:
        Name of the backend to use.

    Returns
    -------
    PolyHNFWithTransformResult
        A Pydantic model with fields:

        ``hermite_normal_form``
            Upper-triangular matrix with monic pivots.
        ``left_transform``
            Invertible m×m matrix U over F_p[x].

        Satisfies: ``left_transform @ matrix = hermite_normal_form``.
    """
    if backend is None:
        backend = _DEFAULTS["hnf_transform"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        return PolyHNFWithTransformResult(
            hermite_normal_form=_to_dense_poly_out(
                _zero_poly_matrix(m.nrows, m.ncols), m.nrows, m.ncols, m.p
            ),
            left_transform=_to_dense_poly_out(
                _identity_poly_matrix(m.nrows), m.nrows, m.nrows, m.p
            ),
        )
    dense = m.to_dense()
    hnf, left = _get_backend(backend).compute_hnf_with_transform(
        dense, m.nrows, m.ncols, m.p
    )
    return PolyHNFWithTransformResult(
        hermite_normal_form=_to_dense_poly_out(hnf, m.nrows, m.ncols, m.p),
        left_transform=_to_dense_poly_out(left, m.nrows, m.nrows, m.p),
    )


def poly_elementary_divisors(
    matrix: Any,
    backend: str | None = None,
) -> PolyElementaryDivisorsResult:
    """Compute the non-zero invariant factors (elementary divisors) over F_p[x].

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`poly_smith_normal_form`).
    backend:
        Name of the backend to use.

    Returns
    -------
    PolyElementaryDivisorsResult
        A Pydantic model with field:

        ``elementary_divisors``
            Monic polynomials in ascending-degree order; same as the
            invariant_factors from :func:`poly_smith_normal_form`.
    """
    if backend is None:
        backend = _DEFAULTS["ed"]
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        return PolyElementaryDivisorsResult(elementary_divisors=[])
    dense = m.to_dense()
    ed = _get_backend(backend).compute_elementary_divisors(dense, m.nrows, m.ncols, m.p)
    return PolyElementaryDivisorsResult(elementary_divisors=ed)
