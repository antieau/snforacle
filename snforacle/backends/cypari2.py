"""Smith normal form backend powered by cypari2 (PARI/GP)."""

from __future__ import annotations

from snforacle.backends.base import SNFBackend


def _pari():
    """Return a cached Pari() singleton (import is deferred so that the
    module can be imported even when cypari2 is not installed)."""
    try:
        import cypari2
    except ImportError as exc:
        raise ImportError(
            "cypari2 is required for the 'cypari2' backend. "
            "Install it with: pip install snforacle[cypari2]"
        ) from exc
    if not hasattr(_pari, "_instance"):
        _pari._instance = cypari2.Pari()
        # Default 8 MB stack is too small for matrices ~1000×1000.
        # Allocate 128 MB to handle matrices up to ~5000×5000.
        _pari._instance.allocatemem(128 * 1024 * 1024, silent=True)
    return _pari._instance


def _gen_matrix_to_list(gen, nrows: int, ncols: int) -> list[list[int]]:
    """Convert a cypari2 Gen matrix to a list of rows of plain Python ints."""
    return [[int(gen[r, c]) for c in range(ncols)] for r in range(nrows)]


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    """Plain-Python integer matrix multiplication."""
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)]
        for i in range(m)
    ]


def _build_snf_matrix(
    inv_factors: list[int], nrows: int, ncols: int
) -> list[list[int]]:
    """Build the standard SNF matrix from non-decreasing invariant factors.

    The factors are placed at (0,0), (1,1), … with all other entries zero.
    """
    mat = [[0] * ncols for _ in range(nrows)]
    for i, d in enumerate(inv_factors):
        if i < nrows and i < ncols:
            mat[i][i] = d
    return mat


def _extract_invariant_factors(pari_mat) -> list[int]:
    """Return the invariant factors of *pari_mat* in non-decreasing order.

    PARI's ``matsnf(flag=0)`` returns a vector whose length equals ``nrows``
    (padding with zeros when the rank is less than the number of rows or
    columns).  We filter out the zeros and sort the result.
    """
    raw = pari_mat.matsnf()
    return sorted(int(raw[i]) for i in range(len(raw)) if int(raw[i]) != 0)


def _permutation_matrices(
    D_pari: list[list[int]],
    D_std: list[list[int]],
    nrows: int,
    ncols: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Find permutation matrices P (m×m) and Q (n×n) such that P @ D_pari @ Q = D_std.

    Both D_pari and D_std are generalized-diagonal integer matrices (the
    SNF before and after normalising to standard form).  P and Q encode the
    bijection between nonzero positions, with zero rows/columns filled in
    arbitrarily.
    """
    from collections import defaultdict

    # Collect nonzero entries, grouped by value.
    def _nonzero_by_value(D):
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for i in range(nrows):
            for j in range(ncols):
                if D[i][j] != 0:
                    groups[D[i][j]].append((i, j))
        return groups

    pari_by_val = _nonzero_by_value(D_pari)
    std_by_val = _nonzero_by_value(D_std)

    # Build the bijection between pari positions and std positions.
    row_map: dict[int, int] = {}  # pari_row -> std_row
    col_map: dict[int, int] = {}  # pari_col -> std_col

    for v, pari_positions in pari_by_val.items():
        for (i_p, j_p), (i_s, j_s) in zip(pari_positions, std_by_val[v]):
            row_map[i_p] = i_s
            col_map[j_p] = j_s

    # Fill in zero rows/columns with the remaining indices.
    unmapped_pari_rows = [i for i in range(nrows) if i not in row_map]
    unmapped_std_rows = [i for i in range(nrows) if i not in row_map.values()]
    for i_p, i_s in zip(unmapped_pari_rows, unmapped_std_rows):
        row_map[i_p] = i_s

    unmapped_pari_cols = [j for j in range(ncols) if j not in col_map]
    unmapped_std_cols = [j for j in range(ncols) if j not in col_map.values()]
    for j_p, j_s in zip(unmapped_pari_cols, unmapped_std_cols):
        col_map[j_p] = j_s

    # Build permutation matrices: P[std_row][pari_row] = 1, Q[std_col][pari_col] = 1.
    P = [[0] * nrows for _ in range(nrows)]
    for i_p, i_s in row_map.items():
        P[i_s][i_p] = 1

    Q = [[0] * ncols for _ in range(ncols)]
    for j_p, j_s in col_map.items():
        Q[j_p][j_s] = 1

    return P, Q


class Cypari2Backend(SNFBackend):
    """Uses PARI/GP's ``matsnf`` function via cypari2.

    Notes
    -----
    PARI's ``matsnf`` uses a non-standard convention: the diagonal of the
    returned matrix may be in decreasing order and right/bottom-aligned for
    non-square matrices.  This backend normalises the output to the standard
    form where the invariant factors appear at positions (0,0), (1,1), …
    in non-decreasing order.
    """

    def _to_pari_matrix(self, matrix: list[list[int]], nrows: int, ncols: int):
        pari = _pari()
        flat = [matrix[r][c] for r in range(nrows) for c in range(ncols)]
        return pari.matrix(nrows, ncols, flat)

    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int]]:
        pari_mat = self._to_pari_matrix(matrix, nrows, ncols)
        inv_factors = _extract_invariant_factors(pari_mat)
        snf_mat = _build_snf_matrix(inv_factors, nrows, ncols)
        return snf_mat, inv_factors

    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int], list[list[int]], list[list[int]]]:
        pari_mat = self._to_pari_matrix(matrix, nrows, ncols)

        # matsnf(flag=1) returns [U, V, D] where U · M · V = D (PARI's D).
        result = pari_mat.matsnf(flag=1)
        U = _gen_matrix_to_list(result[0], nrows, nrows)
        V = _gen_matrix_to_list(result[1], ncols, ncols)
        D_pari = _gen_matrix_to_list(result[2], nrows, ncols)

        # Build standard-form D_std.
        inv_factors = _extract_invariant_factors(pari_mat)
        D_std = _build_snf_matrix(inv_factors, nrows, ncols)

        # Find P, Q such that P @ D_pari @ Q = D_std, then adjust transforms:
        # (P @ U) @ M @ (V @ Q) = P @ D_pari @ Q = D_std.
        P, Q = _permutation_matrices(D_pari, D_std, nrows, ncols)
        U_prime = _mat_mul(P, U)
        V_prime = _mat_mul(V, Q)

        return D_std, inv_factors, U_prime, V_prime
