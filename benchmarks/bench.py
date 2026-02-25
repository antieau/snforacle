"""Benchmarking suite comparing all snforacle backends across matrix sizes.

Usage:
    python benchmarks/bench.py

Results are printed as an ASCII table and saved to benchmarks/results.csv.
"""

from __future__ import annotations

import csv
import random
import shutil
import signal
import statistics
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SIZES = [10, 20, 30, 50, 75]
_BACKENDS = ["cypari2", "flint", "sage", "magma"]
_NO_TRANSFORMS = {"flint"}  # backends that don't support transforms
_TIMEOUT = 120  # seconds
_REPEATS_SMALL = 3   # sizes <= 100
_REPEATS_LARGE = 1   # sizes > 100
_THRESHOLD_REPEATS = 100

# ---------------------------------------------------------------------------
# Matrix generation
# ---------------------------------------------------------------------------

def _make_dense_matrix(n: int, seed: int = 42) -> list[list[int]]:
    """Return an n×n dense matrix with entries uniform in [-100, 100]."""
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        return rng.integers(-100, 101, size=(n, n)).tolist()
    except ImportError:
        rng = random.Random(seed)
        return [[rng.randint(-100, 100) for _ in range(n)] for _ in range(n)]


def _make_sparse_matrix(n: int, seed: int = 42) -> list[list[int]]:
    """Return an n×n sparse matrix with 5% nonzero entries, each ±1."""
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        mask = rng.random(size=(n, n)) < 0.05
        signs = rng.choice([-1, 1], size=(n, n))
        mat = (mask * signs).tolist()
        return [[int(v) for v in row] for row in mat]
    except ImportError:
        rng = random.Random(seed)
        total = n * n
        nnz = max(1, int(total * 0.05))
        flat = [0] * total
        positions = rng.sample(range(total), min(nnz, total))
        for pos in positions:
            flat[pos] = rng.choice([-1, 1])
        return [flat[r * n : (r + 1) * n] for r in range(n)]


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------

def _check_available(backend_name: str) -> bool:
    """Return True if the named backend can be instantiated / is on PATH."""
    cli_backends = {"sage", "magma"}
    if backend_name in cli_backends:
        return shutil.which(backend_name) is not None
    # Python backends: attempt import
    try:
        from snforacle.interface import _BACKENDS  # noqa: PLC0415
        return backend_name in _BACKENDS
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutError()


def _time_backend(
    name: str, matrix: list[list[int]], n: int, timeout: int,
    *, transforms: bool = False,
) -> float | str:
    """Time a single SNF call; return elapsed seconds or 'timeout'/'error'."""
    from snforacle import smith_normal_form, smith_normal_form_with_transforms  # noqa: PLC0415

    inp = {
        "format": "dense",
        "nrows": n,
        "ncols": n,
        "entries": matrix,
    }

    fn = smith_normal_form_with_transforms if transforms else smith_normal_form

    # Use signal.alarm on Unix for Python backends; CLI backends handle their
    # own timeout via subprocess.TimeoutExpired.
    cli_backends = {"sage", "magma"}
    use_signal = (name not in cli_backends) and hasattr(signal, "SIGALRM")

    try:
        if use_signal:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)
        start = time.perf_counter()
        fn(inp, backend=name)
        elapsed = time.perf_counter() - start
        return elapsed
    except _TimeoutError:
        return "timeout"
    except TimeoutError:
        return "timeout"
    except Exception as exc:
        return f"error: {exc}"
    finally:
        if use_signal:
            signal.alarm(0)


def _run_benchmark(
    name: str, size: int, variant: str, matrix: list[list[int]],
    *, transforms: bool = False,
) -> str:
    """Run timed benchmark(s) for one (backend, size, variant) combination."""
    repeats = _REPEATS_SMALL if size <= _THRESHOLD_REPEATS else _REPEATS_LARGE
    times: list[float] = []

    for _ in range(repeats):
        result = _time_backend(name, matrix, size, _TIMEOUT, transforms=transforms)
        if isinstance(result, str):
            return result  # 'timeout' or 'error: ...'
        times.append(result)

    median = statistics.median(times)
    return f"{median:.4f}"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_table(rows: list[tuple[str, ...]]) -> None:
    headers = ("Backend", "Size", "Variant", "Time (s)")
    col_widths = [len(h) for h in headers]
    all_rows = [headers] + list(rows)
    for row in all_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def _fmt_row(row):
        return "|" + "|".join(f" {str(c):<{w}} " for c, w in zip(row, col_widths)) + "|"

    print(sep)
    print(_fmt_row(headers))
    print(sep)
    for row in rows:
        print(_fmt_row(row))
    print(sep)


def _save_csv(rows: list[tuple[str, ...]], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Backend", "Size", "Variant", "Time (s)"])
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("snforacle benchmark suite")
    print("Checking backend availability...")
    available: dict[str, bool] = {b: _check_available(b) for b in _BACKENDS}
    for name, avail in available.items():
        status = "available" if avail else "unavailable (skipped)"
        print(f"  {name}: {status}")
    print()

    rows: list[tuple[str, ...]] = []

    for size in _SIZES:
        dense_matrix = _make_dense_matrix(size)
        sparse_matrix = _make_sparse_matrix(size)

        for variant, matrix, transforms in [
            ("dense", dense_matrix, False),
            ("sparse", sparse_matrix, False),
            ("dense+transforms", dense_matrix, True),
            ("sparse+transforms", sparse_matrix, True),
        ]:
            for backend in _BACKENDS:
                if not available[backend]:
                    result_str = "N/A"
                elif transforms and backend in _NO_TRANSFORMS:
                    result_str = "N/A"
                else:
                    print(f"  {backend:8s} n={size:6d} {variant} ...", end="", flush=True)
                    result_str = _run_benchmark(
                        backend, size, variant, matrix, transforms=transforms,
                    )
                    print(f" {result_str}")
                rows.append((backend, str(size), variant, result_str))

    print()
    _print_table(rows)

    csv_path = Path(__file__).parent / "results.csv"
    _save_csv(rows, csv_path)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
