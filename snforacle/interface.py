"""Public API for snforacle.

The two main entry points are:

``smith_normal_form(matrix, backend)``
    Returns the SNF diagonal matrix and invariant factors.

``smith_normal_form_with_transforms(matrix, backend)``
    Returns the SNF together with the unimodular left and right
    transformation matrices U, V such that U · M · V = SNF.

Both functions accept *matrix* as either:

* a validated ``DenseIntMatrix`` / ``SparseIntMatrix`` Pydantic model, or
* a plain ``dict`` that will be validated against those schemas.

All return types are Pydantic models with a ``.model_dump()`` / ``.model_dump_json()``
method for easy serialisation.
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from snforacle.backends.base import SNFBackend
from snforacle.schema import (
    DenseIntMatrix,
    IntMatrix,
    SNFResult,
    SNFWithTransformsResult,
    SparseIntMatrix,
)

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: dict[str, type[SNFBackend]] = {}


def _register(name: str, cls: type[SNFBackend]) -> None:
    _BACKENDS[name] = cls


def _get_backend(name: str) -> SNFBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Available backends: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[name]()


# Register bundled backends (import errors are deferred until first use).
def _lazy_register_cypari2() -> None:
    from snforacle.backends.cypari2 import Cypari2Backend

    _register("cypari2", Cypari2Backend)


def _lazy_register_flint() -> None:
    from snforacle.backends.flint import FlintBackend

    _register("flint", FlintBackend)


def _lazy_register_sage() -> None:
    from snforacle.backends.sage import SageBackend

    _register("sage", SageBackend)


def _lazy_register_magma() -> None:
    from snforacle.backends.magma import MagmaBackend

    _register("magma", MagmaBackend)


_lazy_register_cypari2()
_lazy_register_flint()
_lazy_register_sage()
_lazy_register_magma()

# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------

_matrix_adapter: TypeAdapter[DenseIntMatrix | SparseIntMatrix] = TypeAdapter(IntMatrix)  # type: ignore[type-arg]


def _parse_matrix(matrix: Any) -> DenseIntMatrix | SparseIntMatrix:
    if isinstance(matrix, (DenseIntMatrix, SparseIntMatrix)):
        return matrix
    return _matrix_adapter.validate_python(matrix)


def _to_dense_model(
    entries: list[list[int]], nrows: int, ncols: int
) -> DenseIntMatrix:
    return DenseIntMatrix(format="dense", nrows=nrows, ncols=ncols, entries=entries)


_MAX_DENSE_ELEMENTS = 10_000_000  # ~280 MB of Python ints


def _check_dense_size(nrows: int, ncols: int) -> None:
    """Raise ValueError if dense expansion would be unreasonably large."""
    n_elements = nrows * ncols
    if n_elements > _MAX_DENSE_ELEMENTS:
        raise ValueError(
            f"Matrix dimensions {nrows}x{ncols} = {n_elements} elements "
            f"exceed the maximum of {_MAX_DENSE_ELEMENTS}. "
            f"Consider using a smaller matrix or a different tool."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def smith_normal_form(
    matrix: Any,
    backend: str = "cypari2",
) -> SNFResult:
    """Compute the Smith normal form of an integer matrix.

    Parameters
    ----------
    matrix:
        The input matrix as a ``DenseIntMatrix``, ``SparseIntMatrix``, or a
        plain ``dict`` conforming to one of those schemas.
    backend:
        Name of the backend to use.  Currently ``"cypari2"`` is supported.

    Returns
    -------
    SNFResult
        A Pydantic model with fields:

        ``smith_normal_form``
            The m×n SNF matrix as a ``DenseIntMatrix``.  The diagonal
            entries d₁ | d₂ | … | dᵣ are in non-decreasing order; all
            other entries are zero.
        ``invariant_factors``
            The sequence ``[d₁, …, dᵣ]``.
    """
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        return SNFResult(
            smith_normal_form=_to_dense_model(m.to_dense(), m.nrows, m.ncols),
            invariant_factors=[],
        )
    _check_dense_size(m.nrows, m.ncols)
    dense = m.to_dense()
    snf_mat, inv = _get_backend(backend).compute_snf(dense, m.nrows, m.ncols)
    return SNFResult(
        smith_normal_form=_to_dense_model(snf_mat, m.nrows, m.ncols),
        invariant_factors=inv,
    )


def smith_normal_form_with_transforms(
    matrix: Any,
    backend: str = "cypari2",
) -> SNFWithTransformsResult:
    """Compute the Smith normal form together with the unimodular transformations.

    Parameters
    ----------
    matrix:
        The input matrix (same accepted types as :func:`smith_normal_form`).
    backend:
        Name of the backend to use.

    Returns
    -------
    SNFWithTransformsResult
        A Pydantic model with fields:

        ``smith_normal_form``
            The m×n SNF matrix as a ``DenseIntMatrix``.
        ``invariant_factors``
            The sequence ``[d₁, …, dᵣ]``.
        ``left_transform``
            Unimodular m×m integer matrix U.
        ``right_transform``
            Unimodular n×n integer matrix V.

        The matrices satisfy ``U @ M @ V = smith_normal_form``.
    """
    m = _parse_matrix(matrix)
    if m.nrows == 0 or m.ncols == 0:
        def _identity(n):
            return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return SNFWithTransformsResult(
            smith_normal_form=_to_dense_model(m.to_dense(), m.nrows, m.ncols),
            invariant_factors=[],
            left_transform=_to_dense_model(_identity(m.nrows), m.nrows, m.nrows),
            right_transform=_to_dense_model(_identity(m.ncols), m.ncols, m.ncols),
        )
    _check_dense_size(m.nrows, m.ncols)
    dense = m.to_dense()
    snf_mat, inv, left, right = _get_backend(backend).compute_snf_with_transforms(
        dense, m.nrows, m.ncols
    )
    return SNFWithTransformsResult(
        smith_normal_form=_to_dense_model(snf_mat, m.nrows, m.ncols),
        invariant_factors=inv,
        left_transform=_to_dense_model(left, m.nrows, m.nrows),
        right_transform=_to_dense_model(right, m.ncols, m.ncols),
    )
