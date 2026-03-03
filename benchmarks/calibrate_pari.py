"""Calibrate the minimum PARI stack size for various matrix shapes and L1 norms.

Runs a binary search (doubling then halving) for each (n, l1_per_cell) pair.
Each probe is run in a separate subprocess so a PARI crash cannot kill the
calibrator.  Results are written to snforacle/backends/pari_cache_table.json.

Usage::

    python benchmarks/calibrate_pari.py

The calibration may take 10–60 minutes depending on hardware.  Interrupt at
any time; the table is written after every row so partial results are saved.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

_TABLE_PATH = Path(__file__).parent.parent / "snforacle" / "backends" / "pari_cache_table.json"

_N_SIZES = [10, 25, 50, 100, 200, 500, 1000, 5000, 10000]
_L1_LEVELS = [0.01, 1.0, 100.0, 10_000.0, 1_000_000.0]

_MIN_MB = 4
_MAX_MB = 4096


# ---------------------------------------------------------------------------
# Matrix generation (duplicated here so calibrate_pari.py is self-contained)
# ---------------------------------------------------------------------------

def _make_matrix_with_l1(n: int, l1_target: float, seed: int = 42) -> list[list[int]]:
    """Generate an n×n integer matrix with ∑|entries| ≈ l1_target."""
    l1_per_cell = l1_target / (n * n)
    rng = random.Random(seed)
    if l1_per_cell < 0.5:
        nnz = max(1, int(l1_target))
        flat = [0] * (n * n)
        for pos in rng.sample(range(n * n), min(nnz, n * n)):
            flat[pos] = rng.choice((-1, 1))
    else:
        max_val = max(1, int(2 * l1_per_cell))
        flat = [rng.randint(-max_val, max_val) for _ in range(n * n)]
    return [flat[r * n:(r + 1) * n] for r in range(n)]


# ---------------------------------------------------------------------------
# Subprocess probe
# ---------------------------------------------------------------------------

_PROBE_TEMPLATE = """\
import sys, json
try:
    import cypari2
    pari = cypari2.Pari()
    pari.allocatemem({stack_bytes}, silent=True)
    matrix = {matrix_repr}
    nrows = ncols = {n}
    flat = [matrix[r][c] for r in range(nrows) for c in range(ncols)]
    pari_mat = pari.matrix(nrows, ncols, flat)
    pari_mat.matsnf()
    print("ok")
except Exception as exc:
    print(f"fail: {{exc}}", file=sys.stderr)
    sys.exit(1)
"""


def _probe(n: int, matrix: list[list[int]], stack_mb: int) -> bool:
    """Return True if the PARI SNF computation succeeds with the given stack."""
    stack_bytes = stack_mb * 1024 * 1024
    script = _PROBE_TEMPLATE.format(
        stack_bytes=stack_bytes,
        matrix_repr=repr(matrix),
        n=n,
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Binary search for minimum stack
# ---------------------------------------------------------------------------

def _calibrate_one(n: int, l1_per_cell: float) -> int:
    """Return minimum stack in MB needed for an n×n matrix with given L1/cell."""
    l1_target = l1_per_cell * n * n
    matrix = _make_matrix_with_l1(n, l1_target)

    # Phase 1: find an upper bound by doubling from _MIN_MB.
    upper = _MIN_MB
    while upper <= _MAX_MB:
        if _probe(n, matrix, upper):
            break
        upper *= 2
    else:
        # Even 1 GB isn't enough; return cap.
        return _MAX_MB

    if upper == _MIN_MB:
        return _MIN_MB

    # Phase 2: binary search between upper//2 and upper.
    lo, hi = upper // 2, upper
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if _probe(n, matrix, mid):
            hi = mid
        else:
            lo = mid

    # Add a 25% safety margin and round up to next power of two.
    needed = hi
    safety = max(needed, int(needed * 1.25))
    # Round up to nearest power of two (for cleaner table values).
    result = _MIN_MB
    while result < safety:
        result *= 2
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("PARI stack calibration")
    print(f"Writing results to: {_TABLE_PATH}")
    print()

    entries: list[dict] = []

    for n in _N_SIZES:
        for l1_per_cell in _L1_LEVELS:
            print(f"  n={n:5d}, l1/cell={l1_per_cell:>12g} ... ", end="", flush=True)
            mb = _calibrate_one(n, l1_per_cell)
            print(f"{mb} MB")
            entries.append({"max_n": n, "l1_per_cell": l1_per_cell, "cache_mb": mb})

            # Write incrementally so partial results survive interruption.
            table = {
                "_doc": (
                    "Minimum PARI stack in MB for matrix of size <= max_n with "
                    "L1/cell <= l1_per_cell. Generated by benchmarks/calibrate_pari.py."
                ),
                "entries": entries,
            }
            _TABLE_PATH.write_text(json.dumps(table, indent=2) + "\n")

    print()
    print(f"Calibration complete. Table saved to {_TABLE_PATH}")


if __name__ == "__main__":
    main()
