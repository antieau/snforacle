"""Abstract base class for polynomial-matrix backends over F_p[x]."""

from __future__ import annotations

from abc import ABC, abstractmethod

from snforacle.poly_schema import Poly


class PolyBackend(ABC):
    """Interface that every F_p[x] backend must implement.

    All methods operate on plain Python ``list[list[Poly]]`` matrices where
    ``Poly = list[int]`` is a coefficient list.  The prime ``p`` is passed
    explicitly to each call.
    """

    @abstractmethod
    def compute_snf(
        self,
        matrix: list[list[Poly]],
        nrows: int,
        ncols: int,
        p: int,
    ) -> tuple[list[list[Poly]], list[Poly]]:
        """Return the Smith normal form and monic invariant factors.

        Returns
        -------
        snf_matrix:
            m×n matrix with monic invariant factors on the diagonal,
            zeros elsewhere.
        invariant_factors:
            ``[d_1, ..., d_r]`` — the nonzero diagonal entries, monic,
            in ascending-degree order.
        """

    @abstractmethod
    def compute_snf_with_transforms(
        self,
        matrix: list[list[Poly]],
        nrows: int,
        ncols: int,
        p: int,
    ) -> tuple[list[list[Poly]], list[Poly], list[list[Poly]], list[list[Poly]]]:
        """Return SNF together with invertible left and right transforms.

        Returns
        -------
        snf_matrix, invariant_factors, left_transform, right_transform

        Satisfies: left_transform @ matrix @ right_transform = snf_matrix.
        """

    @abstractmethod
    def compute_hnf(
        self,
        matrix: list[list[Poly]],
        nrows: int,
        ncols: int,
        p: int,
    ) -> tuple[list[list[Poly]]]:
        """Return the row Hermite Normal Form.

        Returns
        -------
        hnf_matrix:
            Upper-triangular matrix with monic pivots; entries above each
            pivot have strictly smaller degree than the pivot.
        """

    @abstractmethod
    def compute_hnf_with_transform(
        self,
        matrix: list[list[Poly]],
        nrows: int,
        ncols: int,
        p: int,
    ) -> tuple[list[list[Poly]], list[list[Poly]]]:
        """Return the row HNF together with the left invertible transform.

        Returns
        -------
        hnf_matrix, left_transform

        Satisfies: left_transform @ matrix = hnf_matrix.
        """

    @abstractmethod
    def compute_elementary_divisors(
        self,
        matrix: list[list[Poly]],
        nrows: int,
        ncols: int,
        p: int,
    ) -> list[Poly]:
        """Return the nonzero invariant factors (elementary divisors).

        Returns
        -------
        elementary_divisors:
            Monic polynomials in ascending-degree order; equivalent to the
            invariant_factors from compute_snf.
        """
