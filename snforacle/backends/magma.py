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

_MAGMA_HNF_TEMPLATE = """\
_m := {nrows};
_n := {ncols};
_A := Matrix(IntegerRing(), _m, _n, [{flat_csv}]);
_H, _U := HermiteForm(_A);
printf "HNF";
for _i in [1.._m] do for _j in [1.._n] do printf " %o", _H[_i,_j]; end for; end for;
printf "\\n";
printf "LEFT";
for _i in [1.._m] do for _j in [1.._m] do printf " %o", _U[_i,_j]; end for; end for;
printf "\\n";
quit;
"""

_MAGMA_ED_TEMPLATE = """\
_m := {nrows};
_n := {ncols};
_A := Matrix(IntegerRing(), _m, _n, [{flat_csv}]);
_divs := ElementaryDivisors(_A);
printf "ED";
for _d in _divs do printf " %o", _d; end for;
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


def _build_magma_script(
    flat: list[int], nrows: int, ncols: int, template: str = _MAGMA_SCRIPT_TEMPLATE
) -> str:
    flat_csv = ", ".join(str(x) for x in flat)
    return template.format(nrows=nrows, ncols=ncols, flat_csv=flat_csv)


def _reshape(flat_ints: list[int], rows: int, cols: int) -> list[list[int]]:
    if len(flat_ints) != rows * cols:
        raise ValueError(
            f"Expected {rows * cols} entries for {rows}x{cols} matrix, "
            f"got {len(flat_ints)}"
        )
    return [flat_ints[r * cols : (r + 1) * cols] for r in range(rows)]


def _safe_parse_ints(tokens: list[str]) -> list[int]:
    """Parse a list of string tokens to integers, handling backslashes from line wrapping.

    When MAGMA output lines are very long, they may wrap, and tokens can have trailing
    backslashes indicating continuation. Strip these before parsing.
    """
    result = []
    for token in tokens:
        # Remove trailing backslash (line continuation marker from MAGMA output wrapping)
        token = token.rstrip('\\').strip()
        # Skip empty tokens (can occur from multiple spaces or line endings)
        if token and token != '':
            try:
                result.append(int(token))
            except ValueError:
                # Skip tokens that can't be parsed as integers
                pass
    return result


def _parse_magma_output(stdout: str, nrows: int, ncols: int) -> dict:
    """Parse SNF, LEFT, RIGHT lines from MAGMA output.

    Handles line wrapping caused by very long output (large matrices or large integers).
    Collects continuation lines by detecting the marker keyword on each logical block.
    When very large numbers span lines, they are properly reconstructed by joining
    the last token of a line ending in backslash with the first token of the next line.
    """
    snf_flat: list[int] | None = None
    left_flat: list[int] | None = None
    right_flat: list[int] | None = None

    # Reconstruct logical lines by collecting continuation lines
    # A new logical line starts when we see SNF, LEFT, or RIGHT markers
    current_block_name = None
    current_block_values = []
    last_token_incomplete = False  # Track if the previous line ended with an incomplete number
    incomplete_token = ""  # Store the incomplete number part

    all_lines = stdout.splitlines()
    for line_idx, line in enumerate(all_lines):
        line = line.strip()
        if not line:
            continue

        tokens = line.split()

        # If previous line ended with backslash, join tokens
        if last_token_incomplete and tokens:
            # The first token of this line continues the previous number
            tokens[0] = incomplete_token + tokens[0]
            last_token_incomplete = False
            incomplete_token = ""

        # Check if this line ends with backslash (indicates number wrapping)
        ends_with_backslash = line.endswith("\\")

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
            # Remove "SNF" marker and parse
            snf_tokens = line[3:].split()
            # Handle the case where the last token might be incomplete
            if ends_with_backslash and snf_tokens:
                last_token = snf_tokens[-1].rstrip('\\')
                snf_tokens[-1] = last_token  # Update with stripped version
                incomplete_token = last_token  # Store for next line
                snf_tokens = snf_tokens[:-1]  # Remove incomplete token from parsing
                last_token_incomplete = True
            current_block_values = _safe_parse_ints(snf_tokens)

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
            # Remove "LEFT" marker and parse
            left_tokens = line[4:].split()
            # Handle incomplete token
            if ends_with_backslash and left_tokens:
                last_token = left_tokens[-1].rstrip('\\')
                left_tokens[-1] = last_token
                incomplete_token = last_token
                left_tokens = left_tokens[:-1]
                last_token_incomplete = True
            current_block_values = _safe_parse_ints(left_tokens)

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
            # Remove "RIGHT" marker and parse
            right_tokens = line[5:].split()
            # Handle incomplete token
            if ends_with_backslash and right_tokens:
                last_token = right_tokens[-1].rstrip('\\')
                right_tokens[-1] = last_token
                incomplete_token = last_token
                right_tokens = right_tokens[:-1]
                last_token_incomplete = True
            current_block_values = _safe_parse_ints(right_tokens)

        else:
            # Continuation line: append values to current block
            if current_block_name is not None:
                # Handle incomplete token
                if ends_with_backslash and tokens:
                    last_token = tokens[-1].rstrip('\\')
                    tokens[-1] = last_token
                    incomplete_token = last_token
                    tokens = tokens[:-1]
                    last_token_incomplete = True
                current_block_values.extend(_safe_parse_ints(tokens))

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


def _parse_magma_hnf_output(stdout: str, nrows: int, ncols: int) -> dict:
    """Parse HNF, LEFT lines from MAGMA output.

    Handles line wrapping caused by very long output, including numbers split across lines.
    """
    hnf_flat: list[int] | None = None
    left_flat: list[int] | None = None

    current_block_name = None
    current_block_values = []
    last_token_incomplete = False
    incomplete_token = ""

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        tokens = line.split()

        # Handle incomplete token from previous line
        if last_token_incomplete and tokens:
            tokens[0] = incomplete_token + tokens[0]
            last_token_incomplete = False
            incomplete_token = ""

        ends_with_backslash = line.endswith("\\")

        # Check for new block marker
        if line.startswith("HNF"):
            # Save previous block if any
            if current_block_name == "HNF":
                hnf_flat = current_block_values
            elif current_block_name == "LEFT":
                left_flat = current_block_values

            # Start new HNF block
            current_block_name = "HNF"
            hnf_tokens = line[3:].split()
            if ends_with_backslash and hnf_tokens:
                last_token = hnf_tokens[-1].rstrip('\\')
                hnf_tokens[-1] = last_token
                incomplete_token = last_token
                hnf_tokens = hnf_tokens[:-1]
                last_token_incomplete = True
            current_block_values = _safe_parse_ints(hnf_tokens)

        elif line.startswith("LEFT"):
            # Save previous block
            if current_block_name == "HNF":
                hnf_flat = current_block_values
            elif current_block_name == "LEFT":
                left_flat = current_block_values

            # Start new LEFT block
            current_block_name = "LEFT"
            left_tokens = line[4:].split()
            if ends_with_backslash and left_tokens:
                last_token = left_tokens[-1].rstrip('\\')
                left_tokens[-1] = last_token
                incomplete_token = last_token
                left_tokens = left_tokens[:-1]
                last_token_incomplete = True
            current_block_values = _safe_parse_ints(left_tokens)

        else:
            # Continuation line: append values to current block
            if current_block_name is not None:
                if ends_with_backslash and tokens:
                    last_token = tokens[-1].rstrip('\\')
                    tokens[-1] = last_token
                    incomplete_token = last_token
                    tokens = tokens[:-1]
                    last_token_incomplete = True
                current_block_values.extend(_safe_parse_ints(tokens))

    # Save the last block
    if current_block_name == "HNF":
        hnf_flat = current_block_values
    elif current_block_name == "LEFT":
        left_flat = current_block_values

    if hnf_flat is None or left_flat is None:
        raise ValueError(
            f"Could not parse MAGMA HNF output. Expected HNF/LEFT lines.\n"
            f"stdout was:\n{stdout}"
        )

    return {
        "hnf": _reshape(hnf_flat, nrows, ncols),
        "left": _reshape(left_flat, nrows, nrows),
    }


def _parse_magma_ed_output(stdout: str) -> dict:
    """Parse ED line from MAGMA output, handling line wrapping and numbers split across lines."""
    ed_flat: list[int] | None = None
    last_token_incomplete = False
    incomplete_token = ""

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        tokens = line.split()

        # Handle incomplete token from previous line
        if last_token_incomplete and tokens:
            tokens[0] = incomplete_token + tokens[0]
            last_token_incomplete = False
            incomplete_token = ""

        ends_with_backslash = line.endswith("\\")

        if line.startswith("ED"):
            ed_tokens = line[2:].split()
            if ends_with_backslash and ed_tokens:
                last_token = ed_tokens[-1].rstrip('\\')
                ed_tokens[-1] = last_token
                incomplete_token = last_token
                ed_tokens = ed_tokens[:-1]
                last_token_incomplete = True
            ed_flat = _safe_parse_ints(ed_tokens)
        elif ed_flat is not None:
            # Continuation line: append more values
            if ends_with_backslash and tokens:
                last_token = tokens[-1].rstrip('\\')
                tokens[-1] = last_token
                incomplete_token = last_token
                tokens = tokens[:-1]
                last_token_incomplete = True
            ed_flat.extend(_safe_parse_ints(tokens))

    if ed_flat is not None:
        return {"elementary_divisors": ed_flat}

    raise ValueError(f"Could not parse MAGMA ED output. stdout was:\n{stdout}")


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

    def _run(
        self,
        matrix: list[list[int]],
        nrows: int,
        ncols: int,
        template: str = _MAGMA_SCRIPT_TEMPLATE,
    ) -> dict:
        if nrows * ncols > _SIZE_GUARD:
            raise ValueError(
                f"Matrix is too large for the MAGMA backend: {nrows}×{ncols} = "
                f"{nrows * ncols} elements exceeds the limit of {_SIZE_GUARD}. "
                "Embedding this many integers inline in a MAGMA script is impractical."
            )
        flat = [matrix[r][c] for r in range(nrows) for c in range(ncols)]
        script_text = _build_magma_script(flat, nrows, ncols, template=template)
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
                if template == _MAGMA_SCRIPT_TEMPLATE:
                    return _parse_magma_output(result.stdout, nrows, ncols)
                elif template == _MAGMA_HNF_TEMPLATE:
                    return _parse_magma_hnf_output(result.stdout, nrows, ncols)
                elif template == _MAGMA_ED_TEMPLATE:
                    return _parse_magma_ed_output(result.stdout)
                else:
                    raise ValueError(f"Unknown MAGMA template: {template}")
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

    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]]]:
        data = self._run(matrix, nrows, ncols, template=_MAGMA_HNF_TEMPLATE)
        return (data["hnf"],)

    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        data = self._run(matrix, nrows, ncols, template=_MAGMA_HNF_TEMPLATE)
        return data["hnf"], data["left"]

    def compute_elementary_divisors(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> list[int]:
        data = self._run(matrix, nrows, ncols, template=_MAGMA_ED_TEMPLATE)
        return data["elementary_divisors"]
