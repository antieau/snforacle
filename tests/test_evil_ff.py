"""Evil edge-case tests for finite-field backends.

Tests all available FF backends against a range of adversarial inputs:
- zero matrices, identity matrices, 1×1 matrices
- rank-0, rank-1, full-rank, rank-deficient
- non-square (wide and tall)
- p=2 (characteristic 2, binary)
- large prime p
- repeated rows/columns
- all-ones, alternating signs (mod p)

All backends that are available are tested. Missing binaries (sage/magma) and
missing python packages (flint) are automatically skipped per test.
"""

from __future__ import annotations

import pytest

from snforacle.backends.pure_python_ff import (
    PurePythonFFBackend,
    mat_mul_ff,
)

# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------

PURE_PYTHON = PurePythonFFBackend()


def _get_flint():
    try:
        from snforacle.backends.flint_ff import FlintFFBackend
        return FlintFFBackend()
    except ImportError:
        return None


def _get_sage():
    import shutil
    if not shutil.which("sage"):
        return None
    try:
        from snforacle.backends.sage_ff import SageFFBackend
        return SageFFBackend()
    except Exception:
        return None


def _get_magma():
    import shutil
    if not shutil.which("magma"):
        return None
    try:
        from snforacle.backends.magma_ff import MagmaFFBackend
        return MagmaFFBackend()
    except Exception:
        return None


def _all_backends():
    backends = {"pure_python": PURE_PYTHON}
    b = _get_flint()
    if b:
        backends["flint"] = b
    b = _get_sage()
    if b:
        backends["sage"] = b
    b = _get_magma()
    if b:
        backends["magma"] = b
    return backends


ALL_BACKENDS = _all_backends()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def _zero(nrows: int, ncols: int) -> list[list[int]]:
    return [[0] * ncols for _ in range(nrows)]


def _is_snf(snf: list[list[int]], nrows: int, ncols: int, rank: int) -> bool:
    for i in range(nrows):
        for j in range(ncols):
            expected = 1 if (i == j and i < rank) else 0
            if snf[i][j] != expected:
                return False
    return True


def _is_rref(H: list[list[int]], nrows: int, ncols: int) -> bool:
    pivot_col = -1
    for i in range(nrows):
        pc = next((j for j in range(ncols) if H[i][j] != 0), None)
        if pc is None:
            # All remaining rows must be zero
            return all(H[ii][j] == 0 for ii in range(i + 1, nrows) for j in range(ncols))
        if pc <= pivot_col or H[i][pc] != 1:
            return False
        if any(H[ii][pc] != 0 for ii in range(nrows) if ii != i):
            return False
        pivot_col = pc
    return True


# ---------------------------------------------------------------------------
# Evil test cases
# ---------------------------------------------------------------------------

EVIL_CASES: list[tuple[str, list[list[int]], int, int, int, int]] = [
    # (name, matrix, nrows, ncols, p, expected_rank)
    ("zero_1x1_p2",         [[0]],                                           1, 1, 2, 0),
    ("unit_1x1_p2",         [[1]],                                           1, 1, 2, 1),
    ("unit_1x1_p7",         [[3]],                                           1, 1, 7, 1),
    ("zero_2x2_p3",         [[0, 0], [0, 0]],                                2, 2, 3, 0),
    ("identity_2x2_p5",     [[1, 0], [0, 1]],                                2, 2, 5, 2),
    ("rank1_2x2_p5",        [[1, 2], [2, 4]],                                2, 2, 5, 1),
    ("rank1_p2",            [[1, 1], [1, 1]],                                2, 2, 2, 1),
    ("all_ones_3x3_p7",     [[1]*3]*3,                                       3, 3, 7, 1),
    ("full_rank_3x3_p7",    [[1, 2, 3], [4, 5, 6], [7, 8, 10]],             3, 3, 7, 3),
    ("singular_3x3_p5",     [[1, 2, 3], [4, 0, 1], [0, 3, 1]],             3, 3, 5, 2),
    ("wide_2x4_p7",         [[1, 0, 2, 3], [0, 1, 4, 2]],                   2, 4, 7, 2),
    ("tall_4x2_p5",         [[1, 0], [0, 1], [2, 3], [4, 1]],               4, 2, 5, 2),
    ("p2_binary_3x3",       [[1, 1, 0], [0, 1, 1], [1, 0, 1]],             3, 3, 2, 2),
    ("p2_singular",         [[1, 1, 0], [1, 1, 0], [0, 0, 0]],             3, 3, 2, 1),
    ("large_p_p101",        [[1, 2], [3, 4]],                                2, 2, 101, 2),
    ("repeated_rows",       [[1, 2, 3], [1, 2, 3], [1, 2, 3]],             3, 3, 7, 1),
    ("repeated_cols",       [[1, 1, 1], [2, 2, 2], [3, 3, 3]],             3, 3, 5, 1),
    ("upper_triangular_p3", [[1, 2, 1], [0, 1, 2], [0, 0, 1]],             3, 3, 3, 3),
    ("lower_triangular_p5", [[1, 0, 0], [2, 1, 0], [3, 4, 1]],             3, 3, 5, 3),
    ("nonsquare_1x3_p7",    [[1, 2, 3]],                                     1, 3, 7, 1),
    ("nonsquare_3x1_p7",    [[1], [2], [3]],                                 3, 1, 7, 1),
]


@pytest.mark.parametrize("backend_name,backend", list(ALL_BACKENDS.items()))
@pytest.mark.parametrize("name,M,nrows,ncols,p,expected_rank", EVIL_CASES)
def test_snf_rank(backend_name, backend, name, M, nrows, ncols, p, expected_rank):
    snf, rank = backend.compute_snf(M, nrows, ncols, p)
    assert rank == expected_rank, f"[{backend_name}/{name}] rank={rank} != {expected_rank}"
    assert _is_snf(snf, nrows, ncols, rank), f"[{backend_name}/{name}] SNF not in canonical form"


@pytest.mark.parametrize("backend_name,backend", list(ALL_BACKENDS.items()))
@pytest.mark.parametrize("name,M,nrows,ncols,p,expected_rank", EVIL_CASES)
def test_snf_with_transforms(backend_name, backend, name, M, nrows, ncols, p, expected_rank):
    snf, rank, U, V = backend.compute_snf_with_transforms(M, nrows, ncols, p)
    assert rank == expected_rank, f"[{backend_name}/{name}] rank={rank} != {expected_rank}"
    assert _is_snf(snf, nrows, ncols, rank), f"[{backend_name}/{name}] SNF not canonical"
    UMV = mat_mul_ff(mat_mul_ff(U, M, p), V, p)
    assert UMV == snf, f"[{backend_name}/{name}] U @ M @ V ≠ snf"


@pytest.mark.parametrize("backend_name,backend", list(ALL_BACKENDS.items()))
@pytest.mark.parametrize("name,M,nrows,ncols,p,expected_rank", EVIL_CASES)
def test_hnf(backend_name, backend, name, M, nrows, ncols, p, expected_rank):
    (H,) = backend.compute_hnf(M, nrows, ncols, p)
    assert _is_rref(H, nrows, ncols), f"[{backend_name}/{name}] H is not RREF"
    nonzero_rows = sum(1 for row in H if any(v != 0 for v in row))
    assert nonzero_rows == expected_rank, (
        f"[{backend_name}/{name}] RREF nonzero rows={nonzero_rows} != rank={expected_rank}"
    )


@pytest.mark.parametrize("backend_name,backend", list(ALL_BACKENDS.items()))
@pytest.mark.parametrize("name,M,nrows,ncols,p,expected_rank", EVIL_CASES)
def test_hnf_with_transform(backend_name, backend, name, M, nrows, ncols, p, expected_rank):
    H, U = backend.compute_hnf_with_transform(M, nrows, ncols, p)
    assert _is_rref(H, nrows, ncols), f"[{backend_name}/{name}] H not RREF"
    UH = mat_mul_ff(U, M, p)
    assert UH == H, f"[{backend_name}/{name}] U @ M ≠ H"


@pytest.mark.parametrize("backend_name,backend", list(ALL_BACKENDS.items()))
@pytest.mark.parametrize("name,M,nrows,ncols,p,expected_rank", EVIL_CASES)
def test_rank(backend_name, backend, name, M, nrows, ncols, p, expected_rank):
    rank = backend.compute_rank(M, nrows, ncols, p)
    assert rank == expected_rank, f"[{backend_name}/{name}] rank={rank} != {expected_rank}"


# ---------------------------------------------------------------------------
# Cross-backend consistency: all available backends must agree
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,M,nrows,ncols,p,expected_rank", EVIL_CASES)
def test_cross_backend_snf_consistency(name, M, nrows, ncols, p, expected_rank):
    """All backends must produce the same (canonical) SNF."""
    results = {}
    for bname, backend in ALL_BACKENDS.items():
        snf, rank = backend.compute_snf(M, nrows, ncols, p)
        results[bname] = (snf, rank)
    # All ranks must match
    ranks = {r for _, r in results.values()}
    assert len(ranks) == 1, f"[{name}] Backends disagree on rank: {results}"
    # All SNFs must match
    snfs = {str(s) for s, _ in results.values()}
    assert len(snfs) == 1, f"[{name}] Backends disagree on SNF: {results}"


@pytest.mark.parametrize("name,M,nrows,ncols,p,expected_rank", EVIL_CASES)
def test_cross_backend_hnf_consistency(name, M, nrows, ncols, p, expected_rank):
    """All backends must produce the same RREF."""
    results = {}
    for bname, backend in ALL_BACKENDS.items():
        (H,) = backend.compute_hnf(M, nrows, ncols, p)
        results[bname] = H
    hnfs = {str(H) for H in results.values()}
    assert len(hnfs) == 1, f"[{name}] Backends disagree on HNF: {results}"
