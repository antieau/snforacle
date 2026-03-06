"""Pydantic models for polynomial-matrix operations over F_p[x].

Input schemas
-------------
DensePolyMatrix  – a matrix of polynomials given as a full list-of-rows.
SparsePolyMatrix – a matrix of polynomials given as a list of (row, col,
                   coeffs) triples.

Polynomials are encoded as coefficient lists ``[c_0, c_1, ..., c_d]`` where
the polynomial is ``c_0 + c_1*x + ... + c_d*x^d``.  All coefficients must be
integers in ``[0, p-1]``.  The zero polynomial is encoded as ``[]``.

Both matrix types carry a ``p`` field (a prime) that defines the field F_p
over which the polynomial ring is built.  The discriminated union ``PolyMatrix``
can be used wherever either format is accepted.

Output schemas
--------------
PolySNFResult               – the Smith normal form diagonal matrix.
PolySNFWithTransformsResult – SNF plus invertible left and right transforms.
PolyHNFResult               – the row Hermite Normal Form.
PolyHNFWithTransformResult  – HNF plus left unimodular transform.
PolyElementaryDivisorsResult– non-zero invariant factors (monic polynomials).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Polynomial type alias and helpers
# ---------------------------------------------------------------------------

# A polynomial is a list of non-negative integers (coefficients mod p).
# [] represents the zero polynomial.
Poly = list[int]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _validate_poly(poly: Poly, p: int, label: str = "entry") -> None:
    """Raise ValueError if *poly* is not a valid polynomial over F_p."""
    for c in poly:
        if not isinstance(c, int) or c < 0 or c >= p:
            raise ValueError(
                f"Polynomial {label} coefficient {c!r} is not in [0, {p-1}]"
            )
    if poly and poly[-1] == 0:
        raise ValueError(
            f"Polynomial {label} {poly} has a trailing zero coefficient "
            "(leading coefficient must be nonzero for non-zero polynomials)"
        )


# ---------------------------------------------------------------------------
# Sparse entry
# ---------------------------------------------------------------------------

class SparsePolyEntry(BaseModel):
    """A single nonzero entry in a sparse polynomial matrix."""

    row: Annotated[int, Field(ge=0)]
    col: Annotated[int, Field(ge=0)]
    coeffs: Poly  # [] = zero polynomial

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class DensePolyMatrix(BaseModel):
    """A matrix over F_p[x] stored as a full list of rows.

    Each entry is a coefficient list ``[c_0, ..., c_d]`` with all
    coefficients in ``[0, p-1]``.  Trailing zeros are not allowed
    (use ``[]`` for the zero polynomial).
    """

    format: Literal["dense_poly"]
    nrows: Annotated[int, Field(ge=0)]
    ncols: Annotated[int, Field(ge=0)]
    p: Annotated[int, Field(ge=2)]
    entries: list[list[Poly]]

    @field_validator("p")
    @classmethod
    def _check_prime(cls, v: int) -> int:
        if not _is_prime(v):
            raise ValueError(f"p={v} is not prime")
        return v

    @model_validator(mode="after")
    def _check_shape_and_coeffs(self) -> "DensePolyMatrix":
        if len(self.entries) != self.nrows:
            raise ValueError(
                f"entries has {len(self.entries)} rows but nrows={self.nrows}"
            )
        for i, row in enumerate(self.entries):
            if len(row) != self.ncols:
                raise ValueError(
                    f"entries row {i} has {len(row)} columns but ncols={self.ncols}"
                )
            for j, poly in enumerate(row):
                _validate_poly(poly, self.p, label=f"[{i}][{j}]")
        return self

    def to_dense(self) -> list[list[Poly]]:
        return [[list(poly) for poly in row] for row in self.entries]


class SparsePolyMatrix(BaseModel):
    """A matrix over F_p[x] stored as (row, col, coeffs) triples.

    Entries not listed are implicitly the zero polynomial.
    """

    format: Literal["sparse_poly"]
    nrows: Annotated[int, Field(ge=0)]
    ncols: Annotated[int, Field(ge=0)]
    p: Annotated[int, Field(ge=2)]
    entries: list[SparsePolyEntry] = Field(default_factory=list)

    @field_validator("p")
    @classmethod
    def _check_prime(cls, v: int) -> int:
        if not _is_prime(v):
            raise ValueError(f"p={v} is not prime")
        return v

    @model_validator(mode="after")
    def _check_bounds_and_duplicates(self) -> "SparsePolyMatrix":
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
            _validate_poly(e.coeffs, self.p, label=f"({e.row},{e.col})")
        return self

    def to_dense(self) -> list[list[Poly]]:
        mat: list[list[Poly]] = [[[] for _ in range(self.ncols)] for _ in range(self.nrows)]
        for e in self.entries:
            mat[e.row][e.col] = list(e.coeffs)
        return mat


# Discriminated union used as the public input type.
PolyMatrix = Annotated[
    DensePolyMatrix | SparsePolyMatrix,
    Field(discriminator="format"),
]


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class DensePolyMatrixOut(BaseModel):
    """A polynomial matrix returned as output (not validated for primality)."""

    nrows: int
    ncols: int
    p: int
    entries: list[list[Poly]]


class PolySNFResult(BaseModel):
    """The Smith normal form of a polynomial matrix over F_p[x].

    Invariant factors are monic polynomials d_1 | d_2 | ... | d_r satisfying
    the divisibility chain, sorted by degree (ascending) then lexicographically.
    The SNF matrix has these on its diagonal; all other entries are zero.
    """

    smith_normal_form: DensePolyMatrixOut
    invariant_factors: list[Poly]


class PolySNFWithTransformsResult(BaseModel):
    """SNF of a polynomial matrix together with invertible transforms.

    Satisfies: left_transform @ M @ right_transform = smith_normal_form.
    The transforms are invertible matrices over F_p[x] (not necessarily
    unimodular in the polynomial sense, but with nonzero constant determinant).
    """

    smith_normal_form: DensePolyMatrixOut
    invariant_factors: list[Poly]
    left_transform: DensePolyMatrixOut
    right_transform: DensePolyMatrixOut


class PolyHNFResult(BaseModel):
    """Row Hermite Normal Form of a polynomial matrix over F_p[x].

    The HNF is the unique upper-triangular matrix H = U @ M where U is
    invertible over F_p[x], the pivots are monic, and entries above each
    pivot have strictly smaller degree than the pivot.
    """

    hermite_normal_form: DensePolyMatrixOut


class PolyHNFWithTransformResult(BaseModel):
    """Row HNF together with the left invertible transform.

    Satisfies: left_transform @ M = hermite_normal_form.
    """

    hermite_normal_form: DensePolyMatrixOut
    left_transform: DensePolyMatrixOut


class PolyElementaryDivisorsResult(BaseModel):
    """Non-zero invariant factors (elementary divisors) of a polynomial matrix.

    These are the same monic polynomials as the diagonal of the SNF,
    returned in ascending-degree order.
    """

    elementary_divisors: list[Poly]
