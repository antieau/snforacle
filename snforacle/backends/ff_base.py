"""Abstract base class for finite-field matrix backends over F_p."""

from __future__ import annotations

from abc import ABC, abstractmethod


class FFBackend(ABC):
    """Interface that every F_p backend must implement.

    All methods operate on plain Python ``list[list[int]]`` matrices where
    entries are integers in ``[0, p-1]``.  The prime ``p`` is passed
    explicitly to each call.
    """

    @abstractmethod
    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int]:
        """Return the SNF matrix and the rank.

        Returns
        -------
        snf_matrix:
            m×n matrix with 1s at (0,0), …, (r-1,r-1) and 0s elsewhere,
            where r is the rank.
        rank:
            The rank r of the matrix over F_p.
        """

    @abstractmethod
    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int, list[list[int]], list[list[int]]]:
        """Return the SNF together with invertible left and right transforms.

        Returns
        -------
        snf_matrix, rank, left_transform, right_transform

        Satisfies: ``left_transform @ matrix @ right_transform = snf_matrix``.
        All entries are in ``[0, p-1]``.
        """

    @abstractmethod
    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]]]:
        """Return the row HNF (= RREF over a field).

        Returns
        -------
        hnf_matrix:
            The unique reduced row echelon form of *matrix* over F_p.
            Upper-staircase with leading 1 in each nonzero row; all other
            entries in each pivot column are 0.
        """

    @abstractmethod
    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Return the row HNF together with the left invertible transform.

        Returns
        -------
        hnf_matrix, left_transform

        Satisfies: ``left_transform @ matrix = hnf_matrix``.
        """

    @abstractmethod
    def compute_rank(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> int:
        """Return the rank of the matrix over F_p."""
