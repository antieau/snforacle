"""Smith normal form backend powered by python-flint."""

from __future__ import annotations

from snforacle.backends.base import SNFBackend


def _flint():
    """Return the flint module (import is deferred so that the module can be
    imported even when python-flint is not installed)."""
    try:
        import flint
    except ImportError as exc:
        raise ImportError(
            "python-flint is required for the 'flint' backend. "
            "Install it with: pip install snforacle[flint]"
        ) from exc
    return flint


class FlintBackend(SNFBackend):
    """Uses python-flint's ``fmpz_mat.snf()`` to compute the Smith normal form.

    Notes
    -----
    Python-FLINT currently only supports SNF without transformation matrices.
    ``compute_snf_with_transforms`` raises ``NotImplementedError``.
    """

    def __init__(self) -> None:
        _flint()  # fail fast if python-flint is not installed

    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int]]:
        flint = _flint()
        mat = flint.fmpz_mat(matrix)
        snf_mat = mat.snf()
        snf_entries = [
            [int(snf_mat[i, j]) for j in range(ncols)] for i in range(nrows)
        ]
        inv_factors = [
            snf_entries[i][i]
            for i in range(min(nrows, ncols))
            if snf_entries[i][i] != 0
        ]
        return snf_entries, inv_factors

    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int], list[list[int]], list[list[int]]]:
        raise NotImplementedError(
            "Python-FLINT does not currently support SNF with transformation matrices."
        )

    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]]]:
        flint = _flint()
        mat = flint.fmpz_mat(matrix)
        H = mat.hnf()
        hnf_entries = [[int(H[i, j]) for j in range(ncols)] for i in range(nrows)]
        return (hnf_entries,)

    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        raise NotImplementedError(
            "Python-FLINT 0.8.0 does not expose hnf_transform(). "
            "Use the sage or magma backend for HNF with transform."
        )

    def compute_elementary_divisors(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> list[int]:
        # Elementary divisors are the non-zero diagonal entries of SNF
        snf_entries, inv_factors = self.compute_snf(matrix, nrows, ncols)
        return inv_factors
