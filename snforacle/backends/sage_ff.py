"""SageMath CLI backend for matrix operations over F_p."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from snforacle.backends.ff_base import FFBackend



def _check_sage() -> None:
    if not shutil.which("sage"):
        raise RuntimeError(
            "sage binary not found on PATH. Install SageMath to use this backend."
        )


def _run_sage(script: str) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "script.sage"
        path.write_text(script)
        result = subprocess.run(
            ["sage", str(path)], capture_output=True, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Sage subprocess failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr}"
        )
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON found in Sage output:\n{result.stdout}")


def _snf_script(
    matrix: list[list[int]], nrows: int, ncols: int, p: int, transforms: bool
) -> str:
    flat = [matrix[i][j] for i in range(nrows) for j in range(ncols)]
    return textwrap.dedent(f"""\
        import json
        from sage.all import *
        p = {p}
        F = GF(p)
        M = matrix(F, {nrows}, {ncols}, {flat})
        D, U, V = M.smith_form()
        # Normalize diagonal: over F_p every nonzero element is a unit,
        # so SNF should have 1s on the diagonal.  smith_form() may return
        # arbitrary units; scale each nonzero D[i,i] to 1 and adjust U.
        for _i in range(min({nrows}, {ncols})):
            if D[_i, _i] != 0:
                _lc = D[_i, _i]
                D[_i, _i] = D[_i, _i] / _lc
                for _j in range({nrows}):
                    U[_i, _j] = U[_i, _j] / _lc
        def m2l(mat, r, c):
            return [[int(mat[i, j]) for j in range(c)] for i in range(r)]
        rank = int(sum(1 for i in range(min({nrows}, {ncols})) if D[i, i] != 0))
        result = {{"snf": m2l(D, {nrows}, {ncols}), "rank": rank}}
        if {str(transforms).lower()}:
            result["left"] = m2l(U, {nrows}, {nrows})
            result["right"] = m2l(V, {ncols}, {ncols})
        print(json.dumps(result))
    """)


def _hnf_script(
    matrix: list[list[int]], nrows: int, ncols: int, p: int, transform: bool
) -> str:
    flat = [matrix[i][j] for i in range(nrows) for j in range(ncols)]
    return textwrap.dedent(f"""\
        import json
        from sage.all import *
        p = {p}
        F = GF(p)
        M = matrix(F, {nrows}, {ncols}, {flat})
        # Compute RREF via augmented matrix to get the left transform.
        Aug = M.augment(identity_matrix(F, {nrows}))
        Aug.echelonize()
        H = Aug.matrix_from_columns(list(range({ncols})))
        U = Aug.matrix_from_columns(list(range({ncols}, {ncols} + {nrows})))
        def m2l(mat, r, c):
            return [[int(mat[i, j]) for j in range(c)] for i in range(r)]
        result = {{"hnf": m2l(H, {nrows}, {ncols})}}
        if {str(transform).lower()}:
            result["left"] = m2l(U, {nrows}, {nrows})
        print(json.dumps(result))
    """)


class SageFFBackend(FFBackend):
    """Uses SageMath CLI subprocess for F_p matrix operations."""

    def __init__(self) -> None:
        _check_sage()  # fail fast if sage is not on PATH

    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int]:
        out = _run_sage(_snf_script(matrix, nrows, ncols, p, False))
        return out["snf"], out["rank"]

    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int, list[list[int]], list[list[int]]]:
        out = _run_sage(_snf_script(matrix, nrows, ncols, p, True))
        return out["snf"], out["rank"], out["left"], out["right"]

    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]]]:
        out = _run_sage(_hnf_script(matrix, nrows, ncols, p, False))
        return (out["hnf"],)

    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        out = _run_sage(_hnf_script(matrix, nrows, ncols, p, True))
        return out["hnf"], out["left"]

    def compute_rank(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> int:
        out = _run_sage(_snf_script(matrix, nrows, ncols, p, False))
        return out["rank"]
