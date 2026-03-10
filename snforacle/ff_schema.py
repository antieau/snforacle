"""Pydantic models for matrix operations over finite fields F_p.

Input schemas
-------------
DenseFFMatrix  – a matrix over F_p stored as a full list of rows.
SparseFFMatrix – a matrix over F_p stored as (row, col, value) triples.

Entries are integers in ``[0, p-1]``.  Sparse entries must be nonzero (i.e.
in ``[1, p-1]``).

Both types carry a ``p`` field (a prime) and are discriminated by the
``format`` field so they can be used wherever an ``FFMatrix`` is expected.

Output schemas
--------------
FFSNFResult                – the SNF (diag(1,...,1,0,...,0)) and rank.
FFSNFWithTransformsResult  – SNF plus invertible left and right transforms.
FFHNFResult                – row Hermite Normal Form (= RREF over a field).
FFHNFWithTransformResult   – HNF plus left invertible transform.
FFRankResult               – just the rank.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Sparse entry
# ---------------------------------------------------------------------------

class SparseFFEntry(BaseModel):
    """A single nonzero entry in a sparse finite-field matrix."""

    row: Annotated[int, Field(ge=0)]
    col: Annotated[int, Field(ge=0)]
    value: int  # validated in [1, p-1] by the matrix model_validator

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class DenseFFMatrix(BaseModel):
    """A matrix over F_p stored as a full list of rows.

    Each entry is an integer in ``[0, p-1]``.
    """

    format: Literal["dense_ff"]
    nrows: Annotated[int, Field(ge=0)]
    ncols: Annotated[int, Field(ge=0)]
    p: Annotated[int, Field(ge=2)]
    entries: list[list[int]]

    @field_validator("p")
    @classmethod
    def _check_prime(cls, v: int) -> int:
        if not _is_prime(v):
            raise ValueError(f"p={v} is not prime")
        return v

    @model_validator(mode="after")
    def _check_shape_and_range(self) -> "DenseFFMatrix":
        if len(self.entries) != self.nrows:
            raise ValueError(
                f"entries has {len(self.entries)} rows but nrows={self.nrows}"
            )
        for i, row in enumerate(self.entries):
            if len(row) != self.ncols:
                raise ValueError(
                    f"entries row {i} has {len(row)} columns but ncols={self.ncols}"
                )
            for j, v in enumerate(row):
                if v < 0 or v >= self.p:
                    raise ValueError(
                        f"Entry [{i}][{j}]={v} is not in [0, {self.p - 1}]"
                    )
        return self

    def to_dense(self) -> list[list[int]]:
        return [list(row) for row in self.entries]


class SparseFFMatrix(BaseModel):
    """A matrix over F_p stored as (row, col, value) triples.

    Entries not listed are implicitly zero.  Listed entries must be nonzero
    (value in ``[1, p-1]``).  Duplicate (row, col) pairs are not allowed.
    """

    format: Literal["sparse_ff"]
    nrows: Annotated[int, Field(ge=0)]
    ncols: Annotated[int, Field(ge=0)]
    p: Annotated[int, Field(ge=2)]
    entries: list[SparseFFEntry] = Field(default_factory=list)

    @field_validator("p")
    @classmethod
    def _check_prime(cls, v: int) -> int:
        if not _is_prime(v):
            raise ValueError(f"p={v} is not prime")
        return v

    @model_validator(mode="after")
    def _check_bounds_and_duplicates(self) -> "SparseFFMatrix":
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
            if e.value <= 0 or e.value >= self.p:
                raise ValueError(
                    f"Entry value {e.value} is not in [1, {self.p - 1}]"
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
FFMatrix = Annotated[
    DenseFFMatrix | SparseFFMatrix,
    Field(discriminator="format"),
]


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

class DenseFFMatrixOut(BaseModel):
    """A finite-field matrix returned as output (entries in [0, p-1])."""

    nrows: int
    ncols: int
    p: int
    entries: list[list[int]]


class FFSNFResult(BaseModel):
    """The Smith normal form of a matrix over F_p.

    Over a field, every nonzero element is a unit, so invariant factors are
    either 1 (nonzero) or 0.  The SNF is ``diag(1, ..., 1, 0, ..., 0)``
    where the number of 1s equals the rank.

    ``smith_normal_form`` has 1s at positions (0,0), ..., (r-1,r-1) and 0s
    elsewhere.  ``rank`` gives r directly.
    """

    smith_normal_form: DenseFFMatrixOut
    rank: int


class FFSNFWithTransformsResult(BaseModel):
    """The SNF together with invertible left and right transforms over F_p.

    Satisfies: ``left_transform @ M @ right_transform = smith_normal_form``.
    All matrices have entries in ``[0, p-1]``.
    """

    smith_normal_form: DenseFFMatrixOut
    rank: int
    left_transform: DenseFFMatrixOut
    right_transform: DenseFFMatrixOut


class FFHNFResult(BaseModel):
    """Row Hermite Normal Form of a matrix over F_p.

    Over a field the HNF is the unique reduced row echelon form (RREF):
    upper-staircase structure with leading 1 in each nonzero row, and all
    other entries in each pivot column equal to 0.
    """

    hermite_normal_form: DenseFFMatrixOut


class FFHNFWithTransformResult(BaseModel):
    """Row HNF together with the left invertible transform over F_p.

    Satisfies: ``left_transform @ M = hermite_normal_form``.
    """

    hermite_normal_form: DenseFFMatrixOut
    left_transform: DenseFFMatrixOut


class FFRankResult(BaseModel):
    """The rank of a matrix over F_p."""

    rank: int
