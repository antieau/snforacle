"""
Pydantic models for snforacle's JSON interface.

Input schemas
-------------
DenseIntMatrix  – a matrix given as a full list-of-rows.
SparseIntMatrix – a matrix given as a list of (row, col, value) triples.

Both are discriminated by the ``format`` field so they can be used
wherever an ``IntMatrix`` is expected.

Output schemas
--------------
SNFResult             – the Smith normal form diagonal matrix.
SNFWithTransformsResult – SNF plus the unimodular left and right
                          transformation matrices (U · M · V = D).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SparseEntry(BaseModel):
    """A single nonzero entry in a sparse integer matrix."""

    row: Annotated[int, Field(ge=0)]
    col: Annotated[int, Field(ge=0)]
    value: int

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def _check_nonzero(self) -> "SparseEntry":
        if self.value == 0:
            raise ValueError(
                f"Sparse entry at ({self.row}, {self.col}) has value 0. "
                "Omit zero entries from sparse format."
            )
        return self


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class DenseIntMatrix(BaseModel):
    """An integer matrix stored as a full list of rows."""

    format: Literal["dense"]
    nrows: Annotated[int, Field(ge=0)]
    ncols: Annotated[int, Field(ge=0)]
    entries: list[list[int]]

    @model_validator(mode="after")
    def _check_shape(self) -> "DenseIntMatrix":
        if len(self.entries) != self.nrows:
            raise ValueError(
                f"entries has {len(self.entries)} rows but nrows={self.nrows}"
            )
        for i, row in enumerate(self.entries):
            if len(row) != self.ncols:
                raise ValueError(
                    f"entries row {i} has {len(row)} columns but ncols={self.ncols}"
                )
        return self

    def to_dense(self) -> list[list[int]]:
        return [list(row) for row in self.entries]


class SparseIntMatrix(BaseModel):
    """An integer matrix stored as a list of (row, col, value) triples.

    Entries not listed are implicitly zero.  Duplicate (row, col) pairs are
    not allowed; the last value wins only if ``allow_duplicates`` is True
    (default: False).
    """

    format: Literal["sparse"]
    nrows: Annotated[int, Field(ge=0)]
    ncols: Annotated[int, Field(ge=0)]
    entries: list[SparseEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_bounds_and_duplicates(self) -> "SparseIntMatrix":
        seen: set[tuple[int, int]] = set()
        for e in self.entries:
            if e.row >= self.nrows:
                raise ValueError(
                    f"Entry row {e.row} is out of bounds for nrows={self.nrows}"
                )
            if e.col >= self.ncols:
                raise ValueError(
                    f"Entry col {e.col} is out of bounds for ncols={self.ncols}"
                )
            key = (e.row, e.col)
            if key in seen:
                raise ValueError(
                    f"Duplicate entry at position ({e.row}, {e.col})"
                )
            seen.add(key)
        return self

    def to_dense(self) -> list[list[int]]:
        mat = [[0] * self.ncols for _ in range(self.nrows)]
        for e in self.entries:
            mat[e.row][e.col] = e.value
        return mat


# Discriminated union used as the public input type.
IntMatrix = Annotated[
    DenseIntMatrix | SparseIntMatrix,
    Field(discriminator="format"),
]


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class SNFResult(BaseModel):
    """The Smith normal form of an integer matrix.

    ``smith_normal_form`` is always returned as a dense matrix whose
    diagonal contains the invariant factors d₁ | d₂ | … | dᵣ (in
    non-decreasing order) and whose off-diagonal entries are zero.

    ``invariant_factors`` is the same sequence as the diagonal, listed
    without the surrounding zero rows/columns.
    """

    smith_normal_form: DenseIntMatrix
    invariant_factors: list[int]


class SNFWithTransformsResult(BaseModel):
    """The Smith normal form together with unimodular transformation matrices.

    Let M be the input matrix.  The matrices satisfy::

        left_transform · M · right_transform = smith_normal_form

    All three matrices are returned as dense integer matrices.
    """

    smith_normal_form: DenseIntMatrix
    invariant_factors: list[int]
    left_transform: DenseIntMatrix
    right_transform: DenseIntMatrix


class HNFResult(BaseModel):
    """Row Hermite Normal Form of an integer matrix.

    The Hermite Normal Form is the unique upper-triangular matrix H with
    positive pivots (smallest to largest) satisfying H = U·M for some
    unimodular integer matrix U.
    """

    hermite_normal_form: DenseIntMatrix


class HNFWithTransformResult(BaseModel):
    """Row Hermite Normal Form together with the left unimodular transform.

    Let M be the input matrix.  The matrices satisfy::

        left_transform · M = hermite_normal_form

    Both matrices are returned as dense integer matrices.
    """

    hermite_normal_form: DenseIntMatrix
    left_transform: DenseIntMatrix


class ElementaryDivisorsResult(BaseModel):
    """Non-zero invariant factors of an integer matrix.

    These are the same values as the diagonal of the Smith normal form,
    returned in non-decreasing order. They can be computed via a potentially
    faster dedicated path than the full SNF computation.
    """

    elementary_divisors: list[int]
