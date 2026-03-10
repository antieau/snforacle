"""Smith normal form backend powered by SageMath (CLI subprocess)."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from snforacle.backends.base import SNFBackend


# Sage script template — written to a .sage file so Sage's preprocessor runs,
# making matrix(ZZ, ...) available without explicit imports.
_SAGE_SCRIPT_TEMPLATE = """\
import json as _json
with open({input_path!r}) as _f:
    _d = _json.load(_f)
_m, _n = _d['nrows'], _d['ncols']
_M = matrix(ZZ, _m, _n, _d['flat'])
_D, _U, _V = _M.smith_form()
_result = {{
    'snf':   [[int(_D[i,j]) for j in range(_n)] for i in range(_m)],
    'left':  [[int(_U[i,j]) for j in range(_m)] for i in range(_m)],
    'right': [[int(_V[i,j]) for j in range(_n)] for i in range(_n)],
}}
print(_json.dumps(_result))
"""

_SAGE_HNF_TEMPLATE = """\
import json as _json
with open({input_path!r}) as _f:
    _d = _json.load(_f)
_m, _n = _d['nrows'], _d['ncols']
_M = matrix(ZZ, _m, _n, _d['flat'])
_H, _U = _M.hermite_form(transformation=True)
_result = {{
    'hnf':  [[int(_H[i,j]) for j in range(_n)] for i in range(_m)],
    'left': [[int(_U[i,j]) for j in range(_m)] for i in range(_m)],
}}
print(_json.dumps(_result))
"""

_SAGE_ED_TEMPLATE = """\
import json as _json
with open({input_path!r}) as _f:
    _d = _json.load(_f)
_m, _n = _d['nrows'], _d['ncols']
_M = matrix(ZZ, _m, _n, _d['flat'])
_divs = [int(d) for d in _M.elementary_divisors() if d != 0]
_result = {{'elementary_divisors': _divs}}
print(_json.dumps(_result))
"""


def _require_sage() -> str:
    """Return path to the sage binary or raise RuntimeError."""
    sage_bin = shutil.which("sage")
    if sage_bin is None:
        raise RuntimeError(
            "SageMath is required for the 'sage' backend but 'sage' was not found on PATH."
        )
    return sage_bin


def _write_input(flat: list[int], nrows: int, ncols: int, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"nrows": nrows, "ncols": ncols, "flat": flat}, f)


def _build_sage_script(input_path: str, template: str = _SAGE_SCRIPT_TEMPLATE) -> str:
    return template.format(input_path=input_path)


def _parse_sage_output(stdout: str, nrows: int, ncols: int) -> dict:
    """Scan lines for the first one starting with '{' and parse it as JSON."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError(
        f"Could not find JSON output from Sage. stdout was:\n{stdout}"
    )


def _extract_invariant_factors_from_snf(
    snf: list[list[int]], nrows: int, ncols: int
) -> list[int]:
    return [snf[i][i] for i in range(min(nrows, ncols)) if snf[i][i] != 0]


class SageBackend(SNFBackend):
    """Uses SageMath's ``smith_form()`` via a CLI subprocess.

    Notes
    -----
    SageMath is invoked as an external process. The ``sage`` binary must be
    on PATH. Both ``compute_snf`` and ``compute_snf_with_transforms`` are
    supported — Sage always computes all three matrices (D, U, V).
    """

    def __init__(self) -> None:
        self._sage_bin = _require_sage()

    def _run(
        self,
        matrix: list[list[int]],
        nrows: int,
        ncols: int,
        template: str = _SAGE_SCRIPT_TEMPLATE,
    ) -> dict:
        flat = [matrix[r][c] for r in range(nrows) for c in range(ncols)]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = str(Path(tmpdir) / "input.json")
            script_path = str(Path(tmpdir) / "snf_script.sage")
            _write_input(flat, nrows, ncols, input_path)
            script_text = _build_sage_script(input_path, template=template)
            with open(script_path, "w") as f:
                f.write(script_text)
            result = subprocess.run(
                [self._sage_bin, script_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Sage subprocess failed (exit {result.returncode}).\n"
                    f"stderr: {result.stderr}"
                )
            return _parse_sage_output(result.stdout, nrows, ncols)

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
        data = self._run(matrix, nrows, ncols, template=_SAGE_HNF_TEMPLATE)
        return (data["hnf"],)

    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        data = self._run(matrix, nrows, ncols, template=_SAGE_HNF_TEMPLATE)
        return data["hnf"], data["left"]

    def compute_elementary_divisors(
        self, matrix: list[list[int]], nrows: int, ncols: int
    ) -> list[int]:
        data = self._run(matrix, nrows, ncols, template=_SAGE_ED_TEMPLATE)
        return data["elementary_divisors"]
