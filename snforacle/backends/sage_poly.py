"""Sage CLI backend for polynomial matrices over F_p[x]."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap

from snforacle.backends.poly_base import PolyBackend
from snforacle.poly_schema import Poly


def _check_sage() -> None:
    if not shutil.which("sage"):
        raise RuntimeError(
            "sage binary not found on PATH. Install SageMath to use this backend."
        )


def _poly_to_sage(poly: Poly) -> str:
    """Format a coefficient list as a Sage polynomial expression."""
    if not poly:
        return "R(0)"
    terms = []
    for i, c in enumerate(poly):
        if c == 0:
            continue
        if i == 0:
            terms.append(str(c))
        elif i == 1:
            terms.append(f"{c}*x" if c != 1 else "x")
        else:
            terms.append(f"{c}*x^{i}" if c != 1 else f"x^{i}")
    return "R(" + (" + ".join(terms) if terms else "0") + ")"


def _matrix_to_sage(matrix: list[list[Poly]], nrows: int, ncols: int) -> str:
    """Format a polynomial matrix as a Sage matrix literal."""
    rows = []
    for i in range(nrows):
        row = "[" + ", ".join(_poly_to_sage(matrix[i][j]) for j in range(ncols)) + "]"
        rows.append(row)
    return "matrix(R, " + str(nrows) + ", " + str(ncols) + ", [" + ", ".join(rows) + "])"


def _run_sage(script: str) -> str:
    """Run a Sage script and return stdout."""
    _check_sage()
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = tmpdir + "/script.sage"
        with open(script_path, "w") as f:
            f.write(script)
        result = subprocess.run(
            ["sage", script_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Sage subprocess failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr}"
        )
    return result.stdout


def _parse_sage_output(stdout: str) -> dict:
    """Parse JSON output from the Sage script."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON found in Sage output:\n{stdout}")


def _snf_script(matrix: list[list[Poly]], nrows: int, ncols: int, p: int, transforms: bool) -> str:
    mat_str = _matrix_to_sage(matrix, nrows, ncols)
    return textwrap.dedent(f"""\
        import json
        from sage.all import *
        p = {p}
        R = GF(p)['x']; x = R.gen()
        M = {mat_str}
        D, U, V = M.smith_form()
        # Normalize diagonal entries to be monic (divide by leading coefficient)
        for i in range(min({nrows}, {ncols})):
            if D[i,i] != 0:
                lc = D[i,i].leading_coefficient()
                if lc != 1:
                    D[i,i] = D[i,i] / lc
                    for j in range({nrows}):
                        U[i,j] = U[i,j] / lc
        def poly_to_list(f):
            c = list(f)
            if not c or all(v == 0 for v in c):
                return []
            # trim trailing zeros
            while c and c[-1] == 0:
                c.pop()
            return [int(v) for v in c]
        def mat_to_lists(mat, nr, nc):
            return [[poly_to_list(mat[i,j]) for j in range(nc)] for i in range(nr)]
        inv = [poly_to_list(D[i,i]) for i in range(min({nrows},{ncols})) if D[i,i] != 0]
        result = {{
            "snf": mat_to_lists(D, {nrows}, {ncols}),
            "inv": inv,
        }}
        if {str(transforms).lower()}:
            result["left"] = mat_to_lists(U, {nrows}, {nrows})
            result["right"] = mat_to_lists(V, {ncols}, {ncols})
        print(json.dumps(result))
    """)


def _hnf_script(matrix: list[list[Poly]], nrows: int, ncols: int, p: int, transform: bool) -> str:
    mat_str = _matrix_to_sage(matrix, nrows, ncols)
    return textwrap.dedent(f"""\
        import json
        from sage.all import *
        p = {p}
        R = GF(p)['x']; x = R.gen()
        M = {mat_str}
        if {str(transform).lower()}:
            H, U = M.hermite_form(transformation=True)
        else:
            H = M.hermite_form()
        def poly_to_list(f):
            c = list(f)
            if not c or all(v == 0 for v in c):
                return []
            while c and c[-1] == 0:
                c.pop()
            return [int(v) for v in c]
        def mat_to_lists(mat, nr, nc):
            return [[poly_to_list(mat[i,j]) for j in range(nc)] for i in range(nr)]
        result = {{"hnf": mat_to_lists(H, {nrows}, {ncols})}}
        if {str(transform).lower()}:
            result["left"] = mat_to_lists(U, {nrows}, {nrows})
        print(json.dumps(result))
    """)


class SagePolyBackend(PolyBackend):
    """Uses SageMath CLI subprocess for polynomial matrix operations over F_p[x]."""

    def compute_snf(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[Poly]]:
        script = _snf_script(matrix, nrows, ncols, p, transforms=False)
        out = _parse_sage_output(_run_sage(script))
        return out["snf"], out["inv"]

    def compute_snf_with_transforms(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[Poly], list[list[Poly]], list[list[Poly]]]:
        script = _snf_script(matrix, nrows, ncols, p, transforms=True)
        out = _parse_sage_output(_run_sage(script))
        return out["snf"], out["inv"], out["left"], out["right"]

    def compute_hnf(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]]]:
        script = _hnf_script(matrix, nrows, ncols, p, transform=False)
        out = _parse_sage_output(_run_sage(script))
        return (out["hnf"],)

    def compute_hnf_with_transform(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[list[Poly]]]:
        script = _hnf_script(matrix, nrows, ncols, p, transform=True)
        out = _parse_sage_output(_run_sage(script))
        return out["hnf"], out["left"]

    def compute_elementary_divisors(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> list[Poly]:
        snf, inv = self.compute_snf(matrix, nrows, ncols, p)
        return inv
