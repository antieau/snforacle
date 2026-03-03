"""Large-scale SNF benchmark parameterized by L1 norm.

Tests matrices up to 10,000×10,000 across five L1-norm levels.
Only the three fastest in-process/CLI backends are included; sage and
pure_python are excluded as too slow at this scale.

Usage::

    python benchmarks/bench_large.py

Results are printed as an ASCII table and saved to benchmarks/large_results.csv.
"""

from __future__ import annotations

import csv
import random
import shutil
import signal
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SIZES = [100, 200, 500, 1000, 2000, 5000, 10000]
_BACKENDS = ["cypari2", "flint", "magma"]
_TIMEOUT = 600  # 10 minutes per cell
_MAGMA_MAX_ELEMENTS = 10_000_000  # inline script embedding limit

_L1_LEVELS: list[tuple[str, object]] = [
    ("sparse",       lambda n: n),
    ("sparse_big",   lambda n: 100 * n),
    ("dense_small",  lambda n: n * n),
    ("dense_medium", lambda n: 100 * n * n),
    ("dense_large",  lambda n: 10_000 * n * n),
]


# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------

def _make_matrix_with_l1(n: int, l1_target: int, seed: int = 42) -> list[list[int]]:
    """Generate an n×n integer matrix with ∑|entries| ≈ l1_target.

    Parameters
    ----------
    n:
        Matrix dimension.
    l1_target:
        Approximate desired sum of absolute values.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[list[int]]
        An n×n matrix of integers.
    """
    l1_per_cell = l1_target / (n * n)
    rng = random.Random(seed)
    if l1_per_cell < 0.5:
        # Sparse: place ≈l1_target entries of ±1 at random positions.
        nnz = max(1, int(l1_target))
        flat = [0] * (n * n)
        for pos in rng.sample(range(n * n), min(nnz, n * n)):
            flat[pos] = rng.choice((-1, 1))
    else:
        # Dense: entries uniform in [-max_val, max_val].
        max_val = max(1, int(2 * l1_per_cell))
        flat = [rng.randint(-max_val, max_val) for _ in range(n * n)]
    return [flat[r * n:(r + 1) * n] for r in range(n)]


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------

def _check_available(backend: str) -> bool:
    if backend == "magma":
        return shutil.which("magma") is not None
    try:
        __import__(backend if backend != "flint" else "flint")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutError()


def _time_snf(backend: str, matrix: list[list[int]], n: int, timeout: int) -> float | str:
    """Time a single SNF computation; return elapsed seconds or 'timeout'/'error: ...'."""
    from snforacle import smith_normal_form  # noqa: PLC0415

    inp = {"format": "dense", "nrows": n, "ncols": n, "entries": matrix}

    cli_backends = {"sage", "magma"}
    use_signal = (backend not in cli_backends) and hasattr(signal, "SIGALRM")

    try:
        if use_signal:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)
        start = time.perf_counter()
        smith_normal_form(inp, backend=backend)
        return time.perf_counter() - start
    except _TimeoutError:
        return "timeout"
    except TimeoutError:
        return "timeout"
    except Exception as exc:
        return f"error: {exc}"
    finally:
        if use_signal:
            signal.alarm(0)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(rows: list[tuple[str, ...]]) -> None:
    headers = ("Backend", "Size", "L1 Level", "Time (s)")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def _fmt(row):
        return "|" + "|".join(f" {str(c):<{w}} " for c, w in zip(row, col_widths)) + "|"

    print(sep)
    print(_fmt(headers))
    print(sep)
    for row in rows:
        print(_fmt(row))
    print(sep)


def _save_csv(rows: list[tuple[str, ...]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Backend", "Size", "L1 Level", "Time (s)"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_RULE = "-" * 62


def main() -> None:
    print("snforacle large-scale benchmark (L1-norm parameterized)")
    print("Checking backend availability...")
    available = {b: _check_available(b) for b in _BACKENDS}
    for name, avail in available.items():
        print(f"  {name}: {'available' if avail else 'unavailable (skipped)'}")

    rows: list[tuple[str, ...]] = []
    # Track which (backend, size) pairs previously timed out so we skip larger sizes.
    timed_out: dict[str, int] = {}  # backend -> smallest timed-out size

    for size in _SIZES:
        print(f"\n{_RULE}")
        print(f"  n = {size}")
        print(_RULE)

        for l1_name, l1_fn in _L1_LEVELS:
            l1_target = int(l1_fn(size))
            matrix = _make_matrix_with_l1(size, l1_target)

            print(f"\n  {l1_name}  (L1 = {l1_target:.2e})")

            for backend in _BACKENDS:
                # Skip if backend unavailable.
                if not available[backend]:
                    result_str = "N/A"
                # Skip MAGMA for matrices exceeding inline script limit.
                elif backend == "magma" and size * size > _MAGMA_MAX_ELEMENTS:
                    result_str = "N/A (too large for MAGMA)"
                # Skip if this backend timed out at a smaller size.
                elif backend in timed_out and size >= timed_out[backend]:
                    result_str = "skipped (prev timeout)"
                else:
                    print(f"    {backend:8s} ...", end="", flush=True)
                    result = _time_snf(backend, matrix, size, _TIMEOUT)
                    if isinstance(result, float):
                        result_str = f"{result:.4f}"
                    else:
                        result_str = result
                        if "timeout" in result_str and backend not in timed_out:
                            timed_out[backend] = size
                    print(f" {result_str}")

                rows.append((backend, str(size), l1_name, result_str))

    print()
    _print_table(rows)

    csv_path = Path(__file__).parent / "large_results.csv"
    _save_csv(rows, csv_path)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
