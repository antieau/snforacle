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
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SIZES = [10, 20, 30, 50, 75]
_BACKENDS = ["cypari2", "flint", "sage", "magma", "pure_python"]
_NO_TRANSFORMS = {"flint"}  # backends that don't support SNF/HNF transforms
_NO_HNF = {"cypari2"}  # backends that don't support HNF (incompatible convention)
_PURE_PYTHON_MAX_SIZE = 20  # O(n⁴) is too slow above this
_TIMEOUT = 120  # seconds
_REPEATS_SMALL = 3   # sizes <= 100
_REPEATS_LARGE = 1   # sizes > 100
_THRESHOLD_REPEATS = 100

# FF (F_p) benchmark configuration
_FF_SIZES = [50, 100, 200, 500, 1000]
_FF_BACKENDS = ["flint", "sage", "magma", "pure_python"]
_FF_NO_TRANSFORMS = {"flint"}  # delegates to pure_python, so equivalent
_FF_PURE_PYTHON_MAX_SIZE = 100  # O(n³) but Python-speed
_FF_P = 65537  # a large prime typical for applications

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


def _check_ff_available(backend_name: str) -> bool:
    """Return True if the named FF backend can be instantiated / is on PATH."""
    cli_backends = {"sage", "magma"}
    if backend_name in cli_backends:
        return shutil.which(backend_name) is not None
    try:
        from snforacle.ff_interface import _BACKENDS as FF_BACKENDS  # noqa: PLC0415
        return backend_name in FF_BACKENDS
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _TimeoutError()


# ---------------------------------------------------------------------------
# CLI startup overhead measurement
# ---------------------------------------------------------------------------

_CLI_BACKENDS = {"sage", "magma"}

# Trivial scripts that do nothing — measure pure startup + teardown cost.
_NOOP_SCRIPTS: dict[str, str] = {
    "sage": "pass\n",
    "magma": "quit;\n",
}

# Script file extensions expected by each CLI tool.
_SCRIPT_EXT: dict[str, str] = {
    "sage": ".sage",
    "magma": ".m",
}

# Extra CLI flags (MAGMA needs -b to suppress banner).
_EXTRA_FLAGS: dict[str, list[str]] = {
    "sage": [],
    "magma": ["-b"],
}


def _measure_startup_overhead(backend_name: str, repeats: int = 3) -> float:
    """Time a no-op script for a CLI backend, return median elapsed seconds."""
    binary = shutil.which(backend_name)
    if binary is None:
        return 0.0

    script_body = _NOOP_SCRIPTS[backend_name]
    ext = _SCRIPT_EXT[backend_name]
    flags = _EXTRA_FLAGS[backend_name]

    times: list[float] = []
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = str(Path(tmpdir) / f"noop{ext}")
            with open(script_path, "w") as f:
                f.write(script_body)
            start = time.perf_counter()
            subprocess.run(
                [binary] + flags + [script_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            times.append(time.perf_counter() - start)

    return statistics.median(times)


def _make_ff_matrix(n: int, p: int, seed: int = 42) -> list[list[int]]:
    """Return an n×n dense matrix with entries uniform in [0, p-1]."""
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
        return rng.integers(0, p, size=(n, n)).tolist()
    except ImportError:
        rng = random.Random(seed)
        return [[rng.randint(0, p - 1) for _ in range(n)] for _ in range(n)]


def _time_ff_backend(
    name: str, matrix: list[list[int]], n: int, p: int, timeout: int,
    *, mode: str = "snf", transforms: bool = False,
    startup_overhead: float = 0.0,
) -> float | str:
    """Time a single FF computation; return elapsed seconds or 'timeout'/'error'."""
    from snforacle import (  # noqa: PLC0415
        ff_hermite_normal_form,
        ff_hermite_normal_form_with_transform,
        ff_rank,
        ff_smith_normal_form,
        ff_smith_normal_form_with_transforms,
    )

    inp = {"format": "dense_ff", "nrows": n, "ncols": n, "p": p, "entries": matrix}

    if mode == "snf":
        fn = ff_smith_normal_form_with_transforms if transforms else ff_smith_normal_form
    elif mode == "hnf":
        fn = ff_hermite_normal_form_with_transform if transforms else ff_hermite_normal_form
    elif mode == "rank":
        fn = ff_rank
    else:
        raise ValueError(f"Unknown FF mode: {mode}")

    use_signal = hasattr(signal, "SIGALRM")

    try:
        if use_signal:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)
        start = time.perf_counter()
        fn(inp, backend=name)
        elapsed = time.perf_counter() - start
        return max(0.0, elapsed - startup_overhead)
    except (_TimeoutError, TimeoutError, subprocess.TimeoutExpired):
        return "timeout"
    except Exception as exc:
        return f"error: {exc}"
    finally:
        if use_signal:
            signal.alarm(0)


def _run_ff_benchmark(
    name: str, size: int, variant: str, matrix: list[list[int]], p: int,
    *, mode: str = "snf", transforms: bool = False,
    startup_overhead: float = 0.0,
) -> str:
    """Run timed FF benchmark(s) for one (backend, size, variant) combination."""
    repeats = _REPEATS_SMALL if size <= _THRESHOLD_REPEATS else _REPEATS_LARGE
    times: list[float] = []

    for _ in range(repeats):
        result = _time_ff_backend(name, matrix, size, p, _TIMEOUT, mode=mode, transforms=transforms, startup_overhead=startup_overhead)
        if isinstance(result, str):
            return result
        times.append(result)

    median = statistics.median(times)
    return f"{median:.4f}"


def _time_backend(
    name: str, matrix: list[list[int]], n: int, timeout: int,
    *, mode: str = "snf", transforms: bool = False,
    startup_overhead: float = 0.0,
) -> float | str:
    """Time a single computation; return elapsed seconds or 'timeout'/'error'.

    Parameters
    ----------
    mode:
        'snf', 'hnf', or 'ed' (elementary divisors).
    transforms:
        Only used for SNF and HNF modes.
    """
    from snforacle import (  # noqa: PLC0415
        elementary_divisors,
        hermite_normal_form,
        hermite_normal_form_with_transform,
        smith_normal_form,
        smith_normal_form_with_transforms,
    )

    inp = {
        "format": "dense",
        "nrows": n,
        "ncols": n,
        "entries": matrix,
    }

    if mode == "snf":
        fn = smith_normal_form_with_transforms if transforms else smith_normal_form
    elif mode == "hnf":
        fn = hermite_normal_form_with_transform if transforms else hermite_normal_form
    elif mode == "ed":
        fn = elementary_divisors
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Use signal.alarm on Unix for Python backends; CLI backends handle their
    # own timeout via subprocess.TimeoutExpired.
    use_signal = hasattr(signal, "SIGALRM")

    try:
        if use_signal:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)
        start = time.perf_counter()
        fn(inp, backend=name)
        elapsed = time.perf_counter() - start
        return max(0.0, elapsed - startup_overhead)
    except (_TimeoutError, TimeoutError, subprocess.TimeoutExpired):
        return "timeout"
    except Exception as exc:
        return f"error: {exc}"
    finally:
        if use_signal:
            signal.alarm(0)


def _run_benchmark(
    name: str, size: int, variant: str, matrix: list[list[int]],
    *, mode: str = "snf", transforms: bool = False,
    startup_overhead: float = 0.0,
) -> str:
    """Run timed benchmark(s) for one (backend, size, variant) combination."""
    repeats = _REPEATS_SMALL if size <= _THRESHOLD_REPEATS else _REPEATS_LARGE
    times: list[float] = []

    for _ in range(repeats):
        result = _time_backend(name, matrix, size, _TIMEOUT, mode=mode, transforms=transforms, startup_overhead=startup_overhead)
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

_RULE = "-" * 62


def main() -> None:
    print("snforacle benchmark suite")
    print()

    # ------------------------------------------------------------------
    # Integer matrix benchmarks
    # ------------------------------------------------------------------
    print("=== Integer matrices ===")
    print("Checking backend availability...")
    available: dict[str, bool] = {b: _check_available(b) for b in _BACKENDS}
    for name, avail in available.items():
        status = "available" if avail else "unavailable (skipped)"
        print(f"  {name}: {status}")

    # Measure CLI backend startup overhead so we can subtract it from timings.
    startup_overheads: dict[str, float] = {}
    for name in _BACKENDS:
        if name in _CLI_BACKENDS and available.get(name):
            print(f"  Measuring {name} startup overhead...", end="", flush=True)
            overhead = _measure_startup_overhead(name)
            startup_overheads[name] = overhead
            print(f" {overhead:.3f}s")

    rows: list[tuple[str, ...]] = []

    for size in _SIZES:
        dense_matrix = _make_dense_matrix(size)
        sparse_matrix = _make_sparse_matrix(size)

        print(f"\n{_RULE}")
        print(f"  n = {size}")
        print(_RULE)

        for variant, matrix, mode, transforms in [
            ("dense", dense_matrix, "snf", False),
            ("sparse", sparse_matrix, "snf", False),
            ("dense+transform", dense_matrix, "snf", True),
            ("sparse+transform", sparse_matrix, "snf", True),
            ("dense+hnf", dense_matrix, "hnf", False),
            ("dense+hnf+transform", dense_matrix, "hnf", True),
            ("dense+ed", dense_matrix, "ed", False),
        ]:
            print(f"\n  {variant}")
            for backend in _BACKENDS:
                if not available[backend]:
                    result_str = "N/A"
                elif backend == "pure_python" and size > _PURE_PYTHON_MAX_SIZE:
                    result_str = "N/A (too large)"
                elif transforms and backend in _NO_TRANSFORMS:
                    result_str = "N/A"
                elif mode == "hnf" and backend in _NO_HNF:
                    result_str = "N/A"
                else:
                    print(f"    {backend:12s} ...", end="", flush=True)
                    result_str = _run_benchmark(
                        backend, size, variant, matrix, mode=mode, transforms=transforms,
                        startup_overhead=startup_overheads.get(backend, 0.0),
                    )
                    print(f" {result_str}")
                rows.append((backend, str(size), variant, result_str))

    print()
    _print_table(rows)

    csv_path = Path(__file__).parent / "results.csv"
    _save_csv(rows, csv_path)
    print(f"\nResults saved to {csv_path}")

    # ------------------------------------------------------------------
    # Finite field (F_p) matrix benchmarks
    # ------------------------------------------------------------------
    print()
    print("=== Finite field matrices (F_p) ===")
    print(f"p = {_FF_P}")
    print("Checking FF backend availability...")
    ff_available: dict[str, bool] = {b: _check_ff_available(b) for b in _FF_BACKENDS}
    for name, avail in ff_available.items():
        status = "available" if avail else "unavailable (skipped)"
        print(f"  {name}: {status}")

    # Measure CLI backend startup overhead for FF (reuse if already measured).
    ff_startup_overheads: dict[str, float] = {}
    for name in _FF_BACKENDS:
        if name in _CLI_BACKENDS and ff_available.get(name):
            if name in startup_overheads:
                ff_startup_overheads[name] = startup_overheads[name]
            else:
                print(f"  Measuring {name} startup overhead...", end="", flush=True)
                overhead = _measure_startup_overhead(name)
                ff_startup_overheads[name] = overhead
                print(f" {overhead:.3f}s")

    ff_rows: list[tuple[str, ...]] = []

    for size in _FF_SIZES:
        ff_matrix = _make_ff_matrix(size, _FF_P)

        print(f"\n{_RULE}")
        print(f"  n = {size}")
        print(_RULE)

        for variant, mode, transforms in [
            ("snf", "snf", False),
            ("snf+transform", "snf", True),
            ("hnf", "hnf", False),
            ("hnf+transform", "hnf", True),
            ("rank", "rank", False),
        ]:
            print(f"\n  {variant}")
            for backend in _FF_BACKENDS:
                if not ff_available[backend]:
                    result_str = "N/A"
                elif backend == "pure_python" and size > _FF_PURE_PYTHON_MAX_SIZE:
                    result_str = "N/A (too large)"
                elif transforms and backend in _FF_NO_TRANSFORMS:
                    # flint delegates to pure_python for transforms — skip at large sizes
                    if size > _FF_PURE_PYTHON_MAX_SIZE:
                        result_str = "N/A (too large)"
                    else:
                        print(f"    {backend:12s} ...", end="", flush=True)
                        result_str = _run_ff_benchmark(
                            backend, size, variant, ff_matrix, _FF_P,
                            mode=mode, transforms=transforms,
                            startup_overhead=ff_startup_overheads.get(backend, 0.0),
                        )
                        print(f" {result_str}")
                    ff_rows.append((backend, str(size), variant, result_str))
                    continue
                else:
                    print(f"    {backend:12s} ...", end="", flush=True)
                    result_str = _run_ff_benchmark(
                        backend, size, variant, ff_matrix, _FF_P,
                        mode=mode, transforms=transforms,
                        startup_overhead=ff_startup_overheads.get(backend, 0.0),
                    )
                    print(f" {result_str}")
                ff_rows.append((backend, str(size), variant, result_str))

    print()
    _print_table(ff_rows)

    ff_csv_path = Path(__file__).parent / "ff_results.csv"
    _save_csv(ff_rows, ff_csv_path)
    print(f"\nFF results saved to {ff_csv_path}")


if __name__ == "__main__":
    main()
