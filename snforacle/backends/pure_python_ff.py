"""Pure Python backend for matrix operations over F_p.

Reference implementation using direct modular Gaussian elimination.
Suitable for any matrix size where O(n^3) field operations are acceptable.
Unlike the polynomial backends there is no exponential coefficient growth,
so this backend handles moderately large matrices (n up to a few hundred)
without difficulty.
"""

from __future__ import annotations

import copy

from snforacle.backends.ff_base import FFBackend


# ---------------------------------------------------------------------------
# Core algorithms (module-level so FlintFFBackend can import them for the
# transform variants where nmod_mat offers no advantage)
# ---------------------------------------------------------------------------

def _inv(c: int, p: int) -> int:
    """Multiplicative inverse of c mod p (p prime, c != 0)."""
    return pow(c, p - 2, p)


def _snf_with_transforms(
    matrix: list[list[int]], nrows: int, ncols: int, p: int
) -> tuple[list[list[int]], int, list[list[int]], list[list[int]]]:
    """Full SNF with left and right unimodular transforms over F_p.

    Returns (snf_matrix, rank, U, V) satisfying U @ matrix @ V = snf_matrix.
    snf_matrix = diag(1,...,1,0,...,0) with *rank* leading 1s.
    """
    W = [row[:] for row in matrix]
    U = [[1 if i == j else 0 for j in range(nrows)] for i in range(nrows)]
    V = [[1 if i == j else 0 for j in range(ncols)] for i in range(ncols)]

    rank = 0
    for k in range(min(nrows, ncols)):
        # Find a pivot in the bottom-right submatrix.
        pi, pj = -1, -1
        for i in range(k, nrows):
            for j in range(k, ncols):
                if W[i][j] % p != 0:
                    pi, pj = i, j
                    break
            if pi != -1:
                break

        if pi == -1:
            break  # no more pivots; remaining submatrix is zero

        # Swap rows k ↔ pi in W and U.
        W[k], W[pi] = W[pi], W[k]
        U[k], U[pi] = U[pi], U[k]

        # Swap cols k ↔ pj in W and V.
        for i in range(nrows):
            W[i][k], W[i][pj] = W[i][pj], W[i][k]
        for i in range(ncols):
            V[i][k], V[i][pj] = V[i][pj], V[i][k]

        # Scale row k so that W[k][k] = 1; apply same scaling to U[k].
        s = _inv(W[k][k] % p, p)
        W[k] = [c * s % p for c in W[k]]
        U[k] = [c * s % p for c in U[k]]

        # Eliminate all other entries in column k (row operations → update U).
        for i in range(nrows):
            if i != k and W[i][k] % p != 0:
                c = W[i][k]
                W[i] = [(W[i][j] - c * W[k][j]) % p for j in range(ncols)]
                U[i] = [(U[i][j] - c * U[k][j]) % p for j in range(nrows)]

        # Eliminate all other entries in row k (col operations → update V).
        for j in range(ncols):
            if j != k and W[k][j] % p != 0:
                c = W[k][j]
                for i in range(nrows):
                    W[i][j] = (W[i][j] - c * W[i][k]) % p
                for i in range(ncols):
                    V[i][j] = (V[i][j] - c * V[i][k]) % p

        rank += 1

    return W, rank, U, V


def _hnf_with_transform(
    matrix: list[list[int]], nrows: int, ncols: int, p: int
) -> tuple[list[list[int]], list[list[int]]]:
    """RREF with left transform over F_p.

    Returns (H, U) satisfying U @ matrix = H where H is the reduced row
    echelon form.
    """
    W = [row[:] for row in matrix]
    U = [[1 if i == j else 0 for j in range(nrows)] for i in range(nrows)]

    pivot_row = 0
    for j in range(ncols):
        # Find a pivot in column j at or below pivot_row.
        pi = -1
        for i in range(pivot_row, nrows):
            if W[i][j] % p != 0:
                pi = i
                break

        if pi == -1:
            continue  # no pivot in this column; move on

        # Swap rows pivot_row ↔ pi in W and U.
        W[pivot_row], W[pi] = W[pi], W[pivot_row]
        U[pivot_row], U[pi] = U[pi], U[pivot_row]

        # Scale pivot row so that W[pivot_row][j] = 1.
        s = _inv(W[pivot_row][j] % p, p)
        W[pivot_row] = [c * s % p for c in W[pivot_row]]
        U[pivot_row] = [c * s % p for c in U[pivot_row]]

        # Eliminate all other entries in column j.
        for i in range(nrows):
            if i != pivot_row and W[i][j] % p != 0:
                c = W[i][j]
                W[i] = [(W[i][jj] - c * W[pivot_row][jj]) % p for jj in range(ncols)]
                U[i] = [(U[i][jj] - c * U[pivot_row][jj]) % p for jj in range(nrows)]

        pivot_row += 1
        if pivot_row == nrows:
            break

    return W, U


def mat_mul_ff(
    A: list[list[int]], B: list[list[int]], p: int
) -> list[list[int]]:
    """Matrix multiplication over F_p.  Useful for verifying transforms."""
    m = len(A)
    n = len(A[0]) if A else 0
    q = len(B[0]) if B else 0
    return [
        [sum(A[i][k] * B[k][j] for k in range(n)) % p for j in range(q)]
        for i in range(m)
    ]


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class PurePythonFFBackend(FFBackend):
    """Pure Python backend for F_p matrix operations.

    Uses direct modular Gaussian elimination.  No external dependencies.
    Suitable for any matrix size where Python-speed O(n^3) arithmetic is
    acceptable (typically n ≲ 500 before performance becomes a concern).
    """

    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int]:
        snf, rank, _, _ = _snf_with_transforms(matrix, nrows, ncols, p)
        return snf, rank

    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int, list[list[int]], list[list[int]]]:
        return _snf_with_transforms(matrix, nrows, ncols, p)

    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]]]:
        H, _ = _hnf_with_transform(matrix, nrows, ncols, p)
        return (H,)

    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        return _hnf_with_transform(matrix, nrows, ncols, p)

    def compute_rank(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> int:
        _, rank, _, _ = _snf_with_transforms(matrix, nrows, ncols, p)
        return rank
