"""Pure Python SNF backend with zero external dependencies.

Educational reference implementation supporting all operations (SNF, HNF,
elementary divisors) with full transforms. Suitable for correctness testing,
education, and environments without native libraries. Complexity is O(n^4)
for n×n matrices; not recommended for large matrices.
"""

from __future__ import annotations

from snforacle.backends.base import SNFBackend


# ---------------------------------------------------------------------------
# Extended GCD and helpers
# ---------------------------------------------------------------------------


def _xgcd(a: int, b: int) -> tuple[int, int, int]:
    """Iterative extended GCD.

    Returns (g, s, t) where g >= 0, a*s + b*t = g, and gcd(a,b) = g.

    Edge cases:
    - _xgcd(0, 0) = (0, 0, 0)
    - _xgcd(a, 0) = (|a|, sign(a), 0)
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    g, s, t = old_r, old_s, old_t
    if g < 0:
        g, s, t = -g, -s, -t

    return g, s, t


def _identity(n: int) -> list[list[int]]:
    """Return the n×n identity matrix."""
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _copy_matrix(M: list[list[int]]) -> list[list[int]]:
    """Deep copy a matrix."""
    return [row[:] for row in M]


def _apply_row_op(
    W: list[list[int]],
    U: list[list[int]],
    i: int,
    j: int,
    a: int,
    b: int,
    c: int,
    d: int,
    ncols: int,
    m: int,
) -> None:
    """Left-multiply rows i,j of W and U by the 2×2 unimodular matrix [[a,b],[c,d]].

    Modifies W and U in-place.
    - new_W[i] = a*W[i] + b*W[j]
    - new_W[j] = c*W[i] + d*W[j]
    - Same transformation on U.

    Parameters
    ----------
    W, U : matrices to modify in-place
    i, j : row indices (i < j)
    a, b, c, d : 2×2 matrix entries (must have det = ad - bc = 1)
    ncols : number of columns in W
    m : number of columns in U (typically m = nrows of original matrix)
    """
    # Compute new values for both rows
    new_Wi = [a * W[i][col] + b * W[j][col] for col in range(ncols)]
    new_Wj = [c * W[i][col] + d * W[j][col] for col in range(ncols)]

    new_Ui = [a * U[i][col] + b * U[j][col] for col in range(m)]
    new_Uj = [c * U[i][col] + d * U[j][col] for col in range(m)]

    # Update in-place
    W[i], W[j] = new_Wi, new_Wj
    U[i], U[j] = new_Ui, new_Uj


def _apply_col_op(
    W: list[list[int]],
    V: list[list[int]],
    i: int,
    j: int,
    a: int,
    b: int,
    c: int,
    d: int,
    nrows: int,
    n: int,
) -> None:
    """Right-multiply columns i,j of W and V by the 2×2 unimodular matrix [[a,c],[b,d]].

    Modifies W and V in-place.
    - new_W[:,i] = a*W[:,i] + b*W[:,j]
    - new_W[:,j] = c*W[:,i] + d*W[:,j]
    - Same transformation on V.

    Parameters
    ----------
    W, V : matrices to modify in-place
    i, j : column indices (i < j)
    a, b, c, d : 2×2 matrix entries (must have det = ad - cb = 1)
    nrows : number of rows in W
    n : number of rows in V (typically n = ncols of original matrix)
    """
    for r in range(nrows):
        new_Wi = a * W[r][i] + b * W[r][j]
        new_Wj = c * W[r][i] + d * W[r][j]
        W[r][i], W[r][j] = new_Wi, new_Wj

    for r in range(n):
        new_Vi = a * V[r][i] + b * V[r][j]
        new_Vj = c * V[r][i] + d * V[r][j]
        V[r][i], V[r][j] = new_Vi, new_Vj


# ---------------------------------------------------------------------------
# SNF with transforms
# ---------------------------------------------------------------------------


def _snf_with_transforms(
    matrix: list[list[int]], nrows: int, ncols: int
) -> tuple[list[list[int]], list[int], list[list[int]], list[list[int]]]:
    """Compute Smith Normal Form with unimodular transforms.

    Returns (W, invariant_factors, U, V) satisfying U @ matrix @ V = W.

    Algorithm: iterative diagonal reduction with Euclidean refinement for
    each pivot. Maintains unimodular transforms U and V at each step.
    Complexity: O(n^4) for square n×n matrices.
    """
    W = _copy_matrix(matrix)
    U = _identity(nrows)
    V = _identity(ncols)

    for k in range(min(nrows, ncols)):
        # Outer loop: repeat until all entries in W[k+1:, k+1:] are divisible by W[k][k]
        while True:
            # Step 1: Find minimum nonzero |W[i][j]| for i >= k, j >= k
            min_val = None
            pi, pj = None, None
            for i in range(k, nrows):
                for j in range(k, ncols):
                    if W[i][j] != 0:
                        if min_val is None or abs(W[i][j]) < min_val:
                            min_val = abs(W[i][j])
                            pi, pj = i, j

            if pi is None:
                # No nonzero entry in remaining block
                break

            # Step 2: Swap pivot to W[k][k]
            if pi != k:
                W[k], W[pi] = W[pi], W[k]
                U[k], U[pi] = U[pi], U[k]

            if pj != k:
                for r in range(nrows):
                    W[r][k], W[r][pj] = W[r][pj], W[r][k]
                for r in range(ncols):
                    V[r][k], V[r][pj] = V[r][pj], V[r][k]

            # Step 3 & 4: Euclidean reduction loop
            while True:
                changed = False

                # Eliminate to the right of W[k][k]
                for j in range(k + 1, ncols):
                    if W[k][j] == 0:
                        continue
                    g, s, t = _xgcd(W[k][k], W[k][j])
                    if g == 0:  # Both are 0, should not happen here
                        continue
                    q = W[k][j] // g
                    p = W[k][k] // g
                    # Apply col operation: [[s, -q], [t, p]], det = s*p + t*q = 1
                    _apply_col_op(W, V, k, j, s, t, -q, p, nrows, ncols)
                    changed = True

                # Eliminate below W[k][k]
                for i in range(k + 1, nrows):
                    if W[i][k] == 0:
                        continue
                    g, s, t = _xgcd(W[k][k], W[i][k])
                    if g == 0:  # Both are 0, should not happen here
                        continue
                    q = W[i][k] // g
                    p = W[k][k] // g
                    # Apply row operation: [[s, t], [-q, p]], det = s*p - t*(-q) = 1
                    _apply_row_op(W, U, k, i, s, t, -q, p, ncols, nrows)
                    changed = True

                if not changed:
                    break

            # Step 5: Check divisibility
            # If W[k][k] divides all W[i][j] for i,j > k, we're done with this k
            # Otherwise, find a divisibility failure and add that column to column k
            divisibility_ok = True
            for i in range(k + 1, nrows):
                for j in range(k + 1, ncols):
                    if W[i][j] != 0 and W[i][j] % W[k][k] != 0:
                        # Add column j to column k in W and V
                        for r in range(nrows):
                            W[r][k] += W[r][j]
                        for r in range(ncols):
                            V[r][k] += V[r][j]
                        divisibility_ok = False
                        break
                if not divisibility_ok:
                    break

            if divisibility_ok:
                # Step 6: Make W[k][k] positive before moving to next k
                if W[k][k] < 0:
                    for j in range(ncols):
                        W[k][j] = -W[k][j]
                    for j in range(nrows):
                        U[k][j] = -U[k][j]
                break

    # Extract invariant factors: nonzero diagonal entries
    invariant_factors = [W[i][i] for i in range(min(nrows, ncols)) if W[i][i] != 0]

    return W, invariant_factors, U, V


# ---------------------------------------------------------------------------
# HNF with transform
# ---------------------------------------------------------------------------


def _hnf_with_transform(
    matrix: list[list[int]], nrows: int, ncols: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Compute row Hermite Normal Form with left unimodular transform.

    Returns (H, U) satisfying U @ matrix = H.
    H is upper triangular with positive pivots in non-decreasing row order,
    and entries above pivots satisfy 0 <= entry < pivot.

    Algorithm: Column-by-column processing with Euclidean reduction in each
    column, followed by above-pivot reduction using floor division.
    Complexity: O(n^3) for square n×n matrices.
    """
    H = _copy_matrix(matrix)
    U = _identity(nrows)
    pivot_row = 0

    for col in range(ncols):
        if pivot_row >= nrows:
            break

        # Euclidean reduction: reduce all nonzero entries in H[pivot_row:, col] to gcd
        for i in range(pivot_row + 1, nrows):
            if H[i][col] == 0:
                continue
            g, s, t = _xgcd(H[pivot_row][col], H[i][col])
            if g == 0:  # Both are 0
                continue
            q = H[i][col] // g
            p = H[pivot_row][col] // g
            # Apply row operation [[s, t], [-q, p]]
            _apply_row_op(H, U, pivot_row, i, s, t, -q, p, ncols, nrows)

        if H[pivot_row][col] == 0:
            # No pivot in this column; move to next column without advancing pivot_row
            continue

        # Make pivot positive
        if H[pivot_row][col] < 0:
            for j in range(ncols):
                H[pivot_row][j] = -H[pivot_row][j]
            for j in range(nrows):
                U[pivot_row][j] = -U[pivot_row][j]

        pivot = H[pivot_row][col]

        # Reduce entries above the pivot using floor division
        # Python's floor division gives result in [0, pivot) for positive pivot
        for i in range(pivot_row):
            if H[i][col] == 0:
                continue
            q = H[i][col] // pivot
            for j in range(ncols):
                H[i][j] -= q * H[pivot_row][j]
            for j in range(nrows):
                U[i][j] -= q * U[pivot_row][j]

        pivot_row += 1

    return H, U


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class PurePythonBackend(SNFBackend):
    """Pure Python Smith Normal Form backend with zero external dependencies."""

    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int]]:
        """Compute Smith Normal Form and invariant factors."""
        W, inv, _, _ = _snf_with_transforms(matrix, nrows, ncols)
        return W, inv

    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int], list[list[int]], list[list[int]]]:
        """Compute SNF with unimodular left and right transforms."""
        return _snf_with_transforms(matrix, nrows, ncols)

    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]]]:
        """Compute row Hermite Normal Form."""
        H, _ = _hnf_with_transform(matrix, nrows, ncols)
        return (H,)

    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Compute HNF with left unimodular transform."""
        return _hnf_with_transform(matrix, nrows, ncols)

    def compute_elementary_divisors(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> list[int]:
        """Compute elementary divisors (non-zero SNF diagonal entries)."""
        _, inv, _, _ = _snf_with_transforms(matrix, nrows, ncols)
        return inv
