"""Smith normal form backend powered by MAGMA (CLI subprocess)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from snforacle.backends.base import SNFBackend

_TIMEOUT = 120  # seconds

# MAGMA script template — matrix entries are embedded directly as a sequence
# literal (MAGMA has no JSON parser).
_MAGMA_SCRIPT_TEMPLATE = """\
_m := {nrows};
_n := {ncols};
_A := Matrix(IntegerRing(), _m, _n, [{flat_csv}]);
_S, _P, _Q := SmithForm(_A);
printf "SNF";
for _i in [1.._m] do for _j in [1.._n] do printf " %o", _S[_i,_j]; end for; end for;
printf "\\n";
printf "LEFT";
for _i in [1.._m] do for _j in [1.._m] do printf " %o", _P[_i,_j]; end for; end for;
printf "\\n";
printf "RIGHT";
for _i in [1.._n] do for _j in [1.._n] do printf " %o", _Q[_i,_j]; end for; end for;
printf "\\n";
quit;
"""

_SIZE_GUARD = 10_000_000  # max elements to embed inline in the script


def _require_magma() -> str:
    """Return path to the magma binary or raise RuntimeError."""
    magma_bin = shutil.which("magma")
    if magma_bin is None:
        raise RuntimeError(
            "MAGMA is required for the 'magma' backend but 'magma' was not found on PATH."
        )
    return magma_bin


def _build_magma_script(flat: list[int], nrows: int, ncols: int) -> str:
    flat_csv = ", ".join(str(x) for x in flat)
    return _MAGMA_SCRIPT_TEMPLATE.format(
        nrows=nrows, ncols=ncols, flat_csv=flat_csv
    )


def _reshape(flat_ints: list[int], rows: int, cols: int) -> list[list[int]]:
    if len(flat_ints) != rows * cols:
        raise ValueError(
            f"Expected {rows * cols} entries for {rows}x{cols} matrix, "
            f"got {len(flat_ints)}"
        )
    return [flat_ints[r * cols : (r + 1) * cols] for r in range(rows)]


def _parse_magma_output(stdout: str, nrows: int, ncols: int) -> dict:
    """Parse SNF, LEFT, RIGHT lines from MAGMA output.

    Handles line wrapping caused by very long output (large matrices or large integers).
    Collects continuation lines by detecting the marker keyword on each logical block.
    """
    snf_flat: list[int] | None = None
    left_flat: list[int] | None = None
    right_flat: list[int] | None = None

    # Reconstruct logical lines by collecting continuation lines
    # A new logical line starts when we see SNF, LEFT, or RIGHT markers
    current_block_name = None
    current_block_values = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check for new block marker
        if line.startswith("SNF"):
            # Save previous block if any
            if current_block_name == "SNF":
                snf_flat = current_block_values
            elif current_block_name == "LEFT":
                left_flat = current_block_values
            elif current_block_name == "RIGHT":
                right_flat = current_block_values

            # Start new SNF block
            current_block_name = "SNF"
            current_block_values = [int(x) for x in line[3:].split()]

        elif line.startswith("LEFT"):
            # Save previous block
            if current_block_name == "SNF":
                snf_flat = current_block_values
            elif current_block_name == "LEFT":
                left_flat = current_block_values
            elif current_block_name == "RIGHT":
                right_flat = current_block_values

            # Start new LEFT block
            current_block_name = "LEFT"
            current_block_values = [int(x) for x in line[4:].split()]

        elif line.startswith("RIGHT"):
            # Save previous block
            if current_block_name == "SNF":
                snf_flat = current_block_values
            elif current_block_name == "LEFT":
                left_flat = current_block_values
            elif current_block_name == "RIGHT":
                right_flat = current_block_values

            # Start new RIGHT block
            current_block_name = "RIGHT"
            current_block_values = [int(x) for x in line[5:].split()]

        else:
            # Continuation line: append values to current block
            if current_block_name is not None:
                current_block_values.extend([int(x) for x in line.split()])

    # Save the last block
    if current_block_name == "SNF":
        snf_flat = current_block_values
    elif current_block_name == "LEFT":
        left_flat = current_block_values
    elif current_block_name == "RIGHT":
        right_flat = current_block_values

    if snf_flat is None or left_flat is None or right_flat is None:
        raise ValueError(
            f"Could not parse MAGMA output. Expected SNF/LEFT/RIGHT lines.\n"
            f"stdout was:\n{stdout}"
        )

    return {
        "snf": _reshape(snf_flat, nrows, ncols),
        "left": _reshape(left_flat, nrows, nrows),
        "right": _reshape(right_flat, ncols, ncols),
    }


def _extract_invariant_factors_from_snf(
    snf: list[list[int]], nrows: int, ncols: int
) -> list[int]:
    return [snf[i][i] for i in range(min(nrows, ncols)) if snf[i][i] != 0]


class MagmaBackend(SNFBackend):
    """Uses MAGMA's ``SmithForm()`` via a CLI subprocess.

    Notes
    -----
    MAGMA is invoked as an external process. The ``magma`` binary must be on
    PATH. Matrix entries are embedded directly in the script as a MAGMA
    sequence literal, so very large matrices are subject to a size guard of
    ``nrows * ncols > 10_000_000``.
    """

    def __init__(self) -> None:
        self._magma_bin = _require_magma()

    def _run(self, matrix: list[list[int]], nrows: int, ncols: int) -> dict:
        if nrows * ncols > _SIZE_GUARD:
            raise ValueError(
                f"Matrix is too large for the MAGMA backend: {nrows}×{ncols} = "
                f"{nrows * ncols} elements exceeds the limit of {_SIZE_GUARD}. "
                "Embedding this many integers inline in a MAGMA script is impractical."
            )
        flat = [matrix[r][c] for r in range(nrows) for c in range(ncols)]
        script_text = _build_magma_script(flat, nrows, ncols)
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = str(Path(tmpdir) / "snf_script.m")
            try:
                with open(script_path, "w") as f:
                    f.write(script_text)
                result = subprocess.run(
                    [self._magma_bin, "-b", script_path],
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"MAGMA subprocess failed (exit {result.returncode}).\n"
                        f"stderr: {result.stderr}"
                    )
                return _parse_magma_output(result.stdout, nrows, ncols)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"MAGMA did not complete within {_TIMEOUT}s."
                ) from exc

    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int]]:
        data = self._run(matrix, nrows, ncols)
        snf = data["snf"]
        return snf, _extract_invariant_factors_from_snf(snf, nrows, ncols)

    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[int], list[list[int]], list[list[int]]]:
        data = self._run(matrix, nrows, ncols)
        snf = data["snf"]
        inv = _extract_invariant_factors_from_snf(snf, nrows, ncols)
        return snf, inv, data["left"], data["right"]
