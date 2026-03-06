"""Pure Python SNF/HNF backend for matrices over F_p[x].

Reference implementation with zero external dependencies.  Polynomials are
represented as ``list[int]`` coefficient vectors ``[c_0, c_1, ..., c_d]``
(constant term first) with ``c_i in [0, p-1]``.  The zero polynomial is
``[]``.  All non-zero polynomials must have a nonzero leading coefficient.

Complexity: O(n^4 * deg) for n×n matrices where deg is the maximum degree
of any entry.  Suitable for small matrices and correctness testing only.
"""

from __future__ import annotations

import copy

from snforacle.backends.poly_base import PolyBackend
from snforacle.poly_schema import Poly


# ---------------------------------------------------------------------------
# F_p[x] arithmetic
# ---------------------------------------------------------------------------

def _trim(coeffs: list[int]) -> Poly:
    """Remove trailing zeros; return [] for the zero polynomial."""
    c = list(coeffs)
    while c and c[-1] == 0:
        c.pop()
    return c


def _neg(a: Poly, p: int) -> Poly:
    return _trim([(-c) % p for c in a])


def _add(a: Poly, b: Poly, p: int) -> Poly:
    n = max(len(a), len(b))
    result = []
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        result.append((ai + bi) % p)
    return _trim(result)


def _sub(a: Poly, b: Poly, p: int) -> Poly:
    n = max(len(a), len(b))
    result = []
    for i in range(n):
        ai = a[i] if i < len(a) else 0
        bi = b[i] if i < len(b) else 0
        result.append((ai - bi) % p)
    return _trim(result)


def _mul(a: Poly, b: Poly, p: int) -> Poly:
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] = (result[i + j] + ai * bj) % p
    return _trim(result)


def _inv_mod_p(c: int, p: int) -> int:
    """Modular inverse of c mod p (p prime, c != 0)."""
    return pow(c, p - 2, p)


def _make_monic(a: Poly, p: int) -> Poly:
    """Return the monic associate of a (divide by leading coefficient)."""
    if not a:
        return []
    lc_inv = _inv_mod_p(a[-1], p)
    return _trim([c * lc_inv % p for c in a])


def _scale(a: Poly, s: int, p: int) -> Poly:
    """Multiply polynomial a by scalar s mod p."""
    return _trim([c * s % p for c in a])


def _divmod_poly(a: Poly, b: Poly, p: int) -> tuple[Poly, Poly]:
    """Polynomial division: return (q, r) with a = q*b + r, deg(r) < deg(b)."""
    if not b:
        raise ZeroDivisionError("Division by zero polynomial")
    r = list(a)
    db = len(b) - 1
    lc_b_inv = _inv_mod_p(b[-1], p)
    q_coeffs: list[int] = []
    while len(r) - 1 >= db and r:
        deg_diff = len(r) - 1 - db
        coeff = r[-1] * lc_b_inv % p
        # Extend q_coeffs to accommodate this term
        while len(q_coeffs) <= deg_diff:
            q_coeffs.append(0)
        q_coeffs[deg_diff] = (q_coeffs[deg_diff] + coeff) % p
        # Subtract coeff * x^deg_diff * b from r
        for i, bi in enumerate(b):
            r[deg_diff + i] = (r[deg_diff + i] - coeff * bi) % p
        r = _trim(r)
    return _trim(q_coeffs), _trim(r)


def _xgcd(a: Poly, b: Poly, p: int) -> tuple[Poly, Poly, Poly]:
    """Extended GCD: return (g, s, t) with g = s*a + t*b, g monic.

    If both a and b are zero, returns ([], [], []).
    """
    old_r, r = list(a), list(b)
    old_s: Poly = [1]
    s: Poly = []
    old_t: Poly = []
    t: Poly = [1]

    while r:
        q, remainder = _divmod_poly(old_r, r, p)
        old_r, r = r, remainder
        old_s, s = s, _sub(old_s, _mul(q, s, p), p)
        old_t, t = t, _sub(old_t, _mul(q, t, p), p)

    g = old_r
    if not g:
        return [], [], []

    lc_inv = _inv_mod_p(g[-1], p)
    return _scale(g, lc_inv, p), _scale(old_s, lc_inv, p), _scale(old_t, lc_inv, p)


def _poly_deg(a: Poly) -> int:
    """Degree of polynomial; -1 for the zero polynomial."""
    return len(a) - 1


def _divides(a: Poly, b: Poly, p: int) -> bool:
    """True if a divides b in F_p[x] (a != [] required)."""
    _, r = _divmod_poly(b, a, p)
    return not r


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def _zero_poly_matrix(nrows: int, ncols: int) -> list[list[Poly]]:
    return [[[] for _ in range(ncols)] for _ in range(nrows)]


def _identity_poly_matrix(n: int) -> list[list[Poly]]:
    return [[[1] if i == j else [] for j in range(n)] for i in range(n)]


def _poly_mat_mul(A: list[list[Poly]], B: list[list[Poly]], p: int) -> list[list[Poly]]:
    m = len(A)
    k = len(A[0]) if A else 0
    n = len(B[0]) if B and B[0] else 0
    C = _zero_poly_matrix(m, n)
    for i in range(m):
        for j in range(n):
            acc: Poly = []
            for l in range(k):
                acc = _add(acc, _mul(A[i][l], B[l][j], p), p)
            C[i][j] = acc
    return C


def _apply_row_op_poly(
    W: list[list[Poly]],
    U: list[list[Poly]],
    i: int, j: int,
    a: Poly, b: Poly, c: Poly, d: Poly,
    ncols: int, m: int, p: int,
) -> None:
    """Left-multiply rows i,j of W and U by [[a,b],[c,d]]."""
    new_Wi = [_add(_mul(a, W[i][col], p), _mul(b, W[j][col], p), p) for col in range(ncols)]
    new_Wj = [_add(_mul(c, W[i][col], p), _mul(d, W[j][col], p), p) for col in range(ncols)]
    new_Ui = [_add(_mul(a, U[i][col], p), _mul(b, U[j][col], p), p) for col in range(m)]
    new_Uj = [_add(_mul(c, U[i][col], p), _mul(d, U[j][col], p), p) for col in range(m)]
    W[i], W[j] = new_Wi, new_Wj
    U[i], U[j] = new_Ui, new_Uj


def _apply_col_op_poly(
    W: list[list[Poly]],
    V: list[list[Poly]],
    i: int, j: int,
    a: Poly, b: Poly, c: Poly, d: Poly,
    nrows: int, n: int, p: int,
) -> None:
    """Right-multiply columns i,j of W and V by [[a,c],[b,d]]."""
    for r in range(nrows):
        new_Wi = _add(_mul(a, W[r][i], p), _mul(b, W[r][j], p), p)
        new_Wj = _add(_mul(c, W[r][i], p), _mul(d, W[r][j], p), p)
        W[r][i], W[r][j] = new_Wi, new_Wj
    for r in range(n):
        new_Vi = _add(_mul(a, V[r][i], p), _mul(b, V[r][j], p), p)
        new_Vj = _add(_mul(c, V[r][i], p), _mul(d, V[r][j], p), p)
        V[r][i], V[r][j] = new_Vi, new_Vj


# ---------------------------------------------------------------------------
# SNF over F_p[x]
# ---------------------------------------------------------------------------

def _snf_with_transforms_poly(
    matrix: list[list[Poly]], nrows: int, ncols: int, p: int
) -> tuple[list[list[Poly]], list[Poly], list[list[Poly]], list[list[Poly]]]:
    """Compute SNF of matrix over F_p[x] with transforms.

    Returns (W, invariant_factors, U, V) with U @ matrix @ V = W.
    Invariant factors are monic.
    """
    W = copy.deepcopy(matrix)
    U = _identity_poly_matrix(nrows)
    V = _identity_poly_matrix(ncols)

    for k in range(min(nrows, ncols)):
        while True:
            # Find minimum-degree nonzero entry in W[k:, k:]
            min_deg: int | None = None
            pi, pj = k, k
            for i in range(k, nrows):
                for j in range(k, ncols):
                    if W[i][j]:
                        d = _poly_deg(W[i][j])
                        if min_deg is None or d < min_deg:
                            min_deg = d
                            pi, pj = i, j

            if min_deg is None:
                break  # remaining block is zero

            # Swap pivot to (k, k)
            if pi != k:
                W[k], W[pi] = W[pi], W[k]
                U[k], U[pi] = U[pi], U[k]
            if pj != k:
                for r in range(nrows):
                    W[r][k], W[r][pj] = W[r][pj], W[r][k]
                for r in range(ncols):
                    V[r][k], V[r][pj] = V[r][pj], V[r][k]

            # If pivot is a unit (degree-0 nonzero = constant in F_p[x]), use
            # direct scalar elimination to avoid infinite GCD cycling.
            if len(W[k][k]) == 1:
                lc_inv = _inv_mod_p(W[k][k][0], p)
                if W[k][k][0] != 1:
                    for col in range(ncols):
                        W[k][col] = _scale(W[k][col], lc_inv, p)
                    for col in range(nrows):
                        U[k][col] = _scale(U[k][col], lc_inv, p)
                # Zero out row k: col_j -= W[k][j] * col_k (col_k unchanged)
                for j in range(k + 1, ncols):
                    if not W[k][j]:
                        continue
                    coeff = W[k][j]
                    for r in range(nrows):
                        W[r][j] = _sub(W[r][j], _mul(coeff, W[r][k], p), p)
                    for r in range(ncols):
                        V[r][j] = _sub(V[r][j], _mul(coeff, V[r][k], p), p)
                # Zero out col k: row_i -= W[i][k] * row_k (row_k unchanged)
                for i in range(k + 1, nrows):
                    if not W[i][k]:
                        continue
                    coeff = W[i][k]
                    for col in range(ncols):
                        W[i][col] = _sub(W[i][col], _mul(coeff, W[k][col], p), p)
                    for col in range(nrows):
                        U[i][col] = _sub(U[i][col], _mul(coeff, U[k][col], p), p)
                break  # [1] divides everything; divisibility trivially satisfied

            # General case: GCD-based Euclidean reduction (pivot has degree >= 1)
            while True:
                changed = False

                for j in range(k + 1, ncols):
                    if not W[k][j]:
                        continue
                    g, s, t = _xgcd(W[k][k], W[k][j], p)
                    q_right, _ = _divmod_poly(W[k][j], g, p)
                    q_left, _ = _divmod_poly(W[k][k], g, p)
                    _apply_col_op_poly(W, V, k, j, s, t, _neg(q_right, p), q_left, nrows, ncols, p)
                    changed = True

                for i in range(k + 1, nrows):
                    if not W[i][k]:
                        continue
                    g, s, t = _xgcd(W[k][k], W[i][k], p)
                    q_below, _ = _divmod_poly(W[i][k], g, p)
                    q_above, _ = _divmod_poly(W[k][k], g, p)
                    _apply_row_op_poly(W, U, k, i, s, t, _neg(q_below, p), q_above, ncols, nrows, p)
                    changed = True

                if not changed:
                    break

            # Divisibility check
            divisibility_ok = True
            for i in range(k + 1, nrows):
                for j in range(k + 1, ncols):
                    if not W[i][j]:
                        continue
                    _, r = _divmod_poly(W[i][j], W[k][k], p)
                    if r:
                        for r_idx in range(nrows):
                            W[r_idx][k] = _add(W[r_idx][k], W[r_idx][j], p)
                        for r_idx in range(ncols):
                            V[r_idx][k] = _add(V[r_idx][k], V[r_idx][j], p)
                        divisibility_ok = False
                        break
                if not divisibility_ok:
                    break

            if divisibility_ok:
                # Make W[k][k] monic
                if W[k][k] and W[k][k][-1] != 1:
                    lc_inv = _inv_mod_p(W[k][k][-1], p)
                    for col in range(ncols):
                        W[k][col] = _scale(W[k][col], lc_inv, p)
                    for col in range(nrows):
                        U[k][col] = _scale(U[k][col], lc_inv, p)
                break

    inv_factors = [W[i][i] for i in range(min(nrows, ncols)) if W[i][i]]
    return W, inv_factors, U, V


# ---------------------------------------------------------------------------
# HNF over F_p[x]
# ---------------------------------------------------------------------------

def _hnf_with_transform_poly(
    matrix: list[list[Poly]], nrows: int, ncols: int, p: int
) -> tuple[list[list[Poly]], list[list[Poly]]]:
    """Compute row HNF of matrix over F_p[x] with left transform.

    Returns (H, U) with U @ matrix = H.  H is upper triangular with monic
    pivots; entries above each pivot have strictly smaller degree than the pivot.
    """
    H = copy.deepcopy(matrix)
    U = _identity_poly_matrix(nrows)
    pivot_row = 0

    for col in range(ncols):
        if pivot_row >= nrows:
            break

        # Euclidean reduction in this column
        for i in range(pivot_row + 1, nrows):
            if not H[i][col]:
                continue
            g, s, t = _xgcd(H[pivot_row][col], H[i][col], p)
            q_i, _ = _divmod_poly(H[i][col], g, p)
            q_p, _ = _divmod_poly(H[pivot_row][col], g, p)
            _apply_row_op_poly(H, U, pivot_row, i, s, t, _neg(q_i, p), q_p, ncols, nrows, p)

        if not H[pivot_row][col]:
            continue

        # Make pivot monic
        if H[pivot_row][col][-1] != 1:
            lc_inv = _inv_mod_p(H[pivot_row][col][-1], p)
            for j in range(ncols):
                H[pivot_row][j] = _scale(H[pivot_row][j], lc_inv, p)
            for j in range(nrows):
                U[pivot_row][j] = _scale(U[pivot_row][j], lc_inv, p)

        pivot = H[pivot_row][col]

        # Reduce entries above pivot: subtract enough multiples so deg < deg(pivot)
        for i in range(pivot_row):
            if not H[i][col]:
                continue
            q, _ = _divmod_poly(H[i][col], pivot, p)
            if not q:
                continue
            for j in range(ncols):
                H[i][j] = _sub(H[i][j], _mul(q, H[pivot_row][j], p), p)
            for j in range(nrows):
                U[i][j] = _sub(U[i][j], _mul(q, U[pivot_row][j], p), p)

        pivot_row += 1

    return H, U


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------

class PurePythonPolyBackend(PolyBackend):
    """Pure Python F_p[x] backend with zero external dependencies."""

    def compute_snf(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[Poly]]:
        W, inv, _, _ = _snf_with_transforms_poly(matrix, nrows, ncols, p)
        return W, inv

    def compute_snf_with_transforms(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[Poly], list[list[Poly]], list[list[Poly]]]:
        return _snf_with_transforms_poly(matrix, nrows, ncols, p)

    def compute_hnf(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]]]:
        H, _ = _hnf_with_transform_poly(matrix, nrows, ncols, p)
        return (H,)

    def compute_hnf_with_transform(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[list[Poly]]]:
        return _hnf_with_transform_poly(matrix, nrows, ncols, p)

    def compute_elementary_divisors(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> list[Poly]:
        _, inv, _, _ = _snf_with_transforms_poly(matrix, nrows, ncols, p)
        return inv


# ---------------------------------------------------------------------------
# Public helper: polynomial matrix multiplication (for test verification)
# ---------------------------------------------------------------------------

def poly_mat_mul(
    A: list[list[Poly]], B: list[list[Poly]], p: int
) -> list[list[Poly]]:
    """Multiply two polynomial matrices over F_p[x]."""
    return _poly_mat_mul(A, B, p)
