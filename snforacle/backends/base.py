"""Abstract base class for Smith normal form backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class SNFBackend(ABC):
    """Interface that every SNF backend must implement.

    All methods operate on plain Python ``list[list[int]]`` matrices so that
    backends are independent of the serialisation layer.
    """

    @abstractmethod
    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int]]:
        """Return the Smith normal form and invariant factors.

        Parameters
        ----------
        matrix:
            Dense integer matrix as a list of rows.
        nrows, ncols:
            Dimensions of *matrix*.

        Returns
        -------
        snf_matrix:
            The m×n Smith normal form matrix in standard form: diagonal
            entries d₁ | d₂ | … | dᵣ at positions (0,0), (1,1), …,
            (r-1,r-1), all other entries zero.
        invariant_factors:
            The same sequence ``[d₁, d₂, …, dᵣ]`` without surrounding zeros.
        """

    @abstractmethod
    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int], list[list[int]], list[list[int]]]:
        """Return the SNF together with the unimodular transformation matrices.

        Returns
        -------
        snf_matrix:
            The m×n Smith normal form matrix (same convention as above).
        invariant_factors:
            Sequence ``[d₁, …, dᵣ]``.
        left_transform:
            Unimodular m×m integer matrix U.
        right_transform:
            Unimodular n×n integer matrix V.

        The matrices satisfy ``U @ M @ V = snf_matrix``.
        """

    @abstractmethod
    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]]]:
        """Return the row Hermite Normal Form of the matrix.

        Parameters
        ----------
        matrix:
            Dense integer matrix as a list of rows.
        nrows, ncols:
            Dimensions of *matrix*.

        Returns
        -------
        hnf_matrix:
            The m×n Hermite Normal Form matrix: upper triangular with
            positive pivots in non-decreasing order from top-left to
            bottom-right. Entries above pivots satisfy 0 ≤ entry < pivot.
            All other entries are zero.
        """

    @abstractmethod
    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Return the row HNF together with the left unimodular transform.

        Returns
        -------
        hnf_matrix:
            The m×n Hermite Normal Form matrix (same convention as above).
        left_transform:
            Unimodular m×m integer matrix U.

        The matrices satisfy ``U @ M = hnf_matrix``.
        """

    @abstractmethod
    def compute_elementary_divisors(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> list[int]:
        """Return the non-zero invariant factors (elementary divisors).

        Returns
        -------
        elementary_divisors:
            The non-zero diagonal entries of the Smith normal form in
            non-decreasing order. Equivalent to the invariant_factors from
            compute_snf, but potentially computed via a faster dedicated path.
        """
