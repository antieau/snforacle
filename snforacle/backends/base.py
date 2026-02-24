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
