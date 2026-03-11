"""Shared matrix helpers for snforacle tests.

Provides determinant computation and unimodularity / invertibility assertions
for all three matrix domains (ZZ, Fp, Fp[x]).
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Integer (ZZ) determinant — Bareiss fraction-free algorithm O(n^3)
# ---------------------------------------------------------------------------

def det_int(M: list[list[int]]) -> int:
    """Return the exact integer determinant of a square matrix."""
    n = len(M)
    if n == 0:
        return 1
    M = [list(row) for row in M]  # work on a copy
    sign = 1
    prev = 1
    for k in range(n):
        # Partial pivot
        pivot = next((i for i in range(k, n) if M[i][k] != 0), None)
        if pivot is None:
            return 0
        if pivot != k:
            M[k], M[pivot] = M[pivot], M[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                M[i][j] = (M[k][k] * M[i][j] - M[i][k] * M[k][j]) // prev
            M[i][k] = 0
        prev = M[k][k]
    return sign * M[n - 1][n - 1]


def assert_unimodular(M: list[list[int]], label: str = "") -> None:
    """Assert that a square integer matrix is unimodular (|det| == 1)."""
    d = det_int(M)
    assert abs(d) == 1, (
        f"{'[' + label + '] ' if label else ''}transform is not unimodular: det={d}"
    )


# ---------------------------------------------------------------------------
# Finite-field (Fp) determinant — Gaussian elimination mod p, O(n^3)
# ---------------------------------------------------------------------------

def det_ff(M: list[list[int]], p: int) -> int:
    """Return the determinant of a square matrix over F_p."""
    n = len(M)
    if n == 0:
        return 1
    M = [[v % p for v in row] for row in M]
    sign = 1
    for k in range(n):
        pivot = next((i for i in range(k, n) if M[i][k] != 0), None)
        if pivot is None:
            return 0
        if pivot != k:
            M[k], M[pivot] = M[pivot], M[k]
            sign = -sign
        inv_pivot = pow(M[k][k], p - 2, p)  # Fermat's little theorem
        for i in range(k + 1, n):
            if M[i][k] != 0:
                factor = M[i][k] * inv_pivot % p
                for j in range(k, n):
                    M[i][j] = (M[i][j] - factor * M[k][j]) % p
    d = sign
    for k in range(n):
        d = d * M[k][k] % p
    return d % p


def assert_invertible_ff(M: list[list[int]], p: int, label: str = "") -> None:
    """Assert that a square matrix over F_p is invertible (det ≠ 0 mod p)."""
    d = det_ff(M, p)
    assert d != 0, (
        f"{'[' + label + '] ' if label else ''}transform is not invertible over F_{p}: det≡0"
    )


# ---------------------------------------------------------------------------
# Polynomial arithmetic over F_p — for Fp[x] determinants
# ---------------------------------------------------------------------------

Poly = list[int]


def _poly_add(a: Poly, b: Poly, p: int) -> Poly:
    n = max(len(a), len(b))
    result = [((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n)]
    while result and result[-1] == 0:
        result.pop()
    return result


def _poly_sub(a: Poly, b: Poly, p: int) -> Poly:
    n = max(len(a), len(b))
    result = [((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % p for i in range(n)]
    while result and result[-1] == 0:
        result.pop()
    return result


def _poly_mul(a: Poly, b: Poly, p: int) -> Poly:
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] = (result[i + j] + ai * bj) % p
    while result and result[-1] == 0:
        result.pop()
    return result


def det_poly(M: list[list[Poly]], p: int) -> Poly:
    """Return the determinant polynomial of a square matrix over F_p[x].

    Uses cofactor expansion — only practical for small matrices (n ≤ 6).
    """
    n = len(M)
    if n == 0:
        return [1]
    if n == 1:
        return list(M[0][0])
    if n == 2:
        return _poly_sub(
            _poly_mul(M[0][0], M[1][1], p),
            _poly_mul(M[0][1], M[1][0], p),
            p,
        )
    result: Poly = []
    for j in range(n):
        minor = [[M[i][k] for k in range(n) if k != j] for i in range(1, n)]
        cofactor = _poly_mul(M[0][j], det_poly(minor, p), p)
        if j % 2 == 0:
            result = _poly_add(result, cofactor, p)
        else:
            result = _poly_sub(result, cofactor, p)
    return result


def assert_invertible_poly(M: list[list[Poly]], p: int, label: str = "") -> None:
    """Assert that a square matrix over F_p[x] is invertible.

    Invertibility over F_p[x] means the determinant is a nonzero constant
    (an element of F_p*, i.e. a unit of the ring).
    """
    d = det_poly(M, p)
    assert len(d) == 1 and d[0] != 0, (
        f"{'[' + label + '] ' if label else ''}transform is not invertible over F_{p}[x]: "
        f"det={d} (must be a nonzero constant polynomial)"
    )
