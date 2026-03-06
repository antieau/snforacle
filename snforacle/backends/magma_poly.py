"""MAGMA CLI backend for polynomial matrices over F_p[x]."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap

from snforacle.backends.poly_base import PolyBackend
from snforacle.poly_schema import Poly


def _check_magma() -> None:
    if not shutil.which("magma"):
        raise RuntimeError(
            "magma binary not found on PATH. Install MAGMA to use this backend."
        )


def _poly_to_magma(poly: Poly) -> str:
    """Format a coefficient list as a MAGMA polynomial expression."""
    if not poly:
        return "R!0"
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
    return "R!(" + (" + ".join(terms) if terms else "0") + ")"


def _matrix_to_magma(matrix: list[list[Poly]], nrows: int, ncols: int) -> str:
    """Format a polynomial matrix as a MAGMA matrix literal."""
    entries = []
    for i in range(nrows):
        for j in range(ncols):
            entries.append(_poly_to_magma(matrix[i][j]))
    return f"Matrix(R, {nrows}, {ncols}, [{', '.join(entries)}])"


def _parse_poly(token: str) -> Poly:
    """Parse a MAGMA polynomial output token like '2*x^3 + x + 1'."""
    token = token.strip()
    if token in ("0", ""):
        return []
    import re
    # Collect terms: coefficient and exponent
    coeffs: dict[int, int] = {}
    # Normalize: handle e.g. "x^2" (coeff=1), "2*x" (coeff=2, exp=1), "3" (exp=0)
    for m in re.finditer(r'([+-]?\s*\d*)\*?x\^?(\d*)|([+-]?\s*\d+)(?!\s*\*?\s*x)', token):
        if m.group(3) is not None:
            # constant term
            c_str = m.group(3).replace(' ', '')
            c = int(c_str) if c_str not in ('+', '-', '') else (1 if '+' in c_str or c_str == '' else -1)
            coeffs[0] = coeffs.get(0, 0) + c
        else:
            c_str = m.group(1).replace(' ', '')
            e_str = m.group(2).replace(' ', '')
            c = int(c_str) if c_str not in ('+', '', '-', '+', '- ') else (1 if (not c_str or c_str == '+') else -1)
            e = int(e_str) if e_str else 1
            coeffs[e] = coeffs.get(e, 0) + c
    if not coeffs:
        return []
    max_deg = max(coeffs)
    result = [coeffs.get(i, 0) for i in range(max_deg + 1)]
    return _trim(result)


def _trim(coeffs: list[int]) -> Poly:
    c = list(coeffs)
    while c and c[-1] == 0:
        c.pop()
    return c


def _parse_magma_matrix(text: str, nrows: int, ncols: int, p: int) -> list[list[Poly]]:
    """Parse MAGMA matrix output (one entry per line in row-major order)."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    entries: list[Poly] = []
    for line in lines:
        # Each line is one polynomial; reduce coefficients mod p
        poly = _parse_poly(line)
        poly = [c % p for c in poly]
        while poly and poly[-1] == 0:
            poly.pop()
        entries.append(poly)
    # Reshape
    mat = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            idx = i * ncols + j
            row.append(entries[idx] if idx < len(entries) else [])
        mat.append(row)
    return mat


def _run_magma(script: str) -> str:
    _check_magma()
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = tmpdir + "/script.magma"
        with open(script_path, "w") as f:
            f.write(script)
        result = subprocess.run(
            ["magma", "-n", script_path],
            capture_output=True,
            text=True,
            timeout=120,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"MAGMA subprocess failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr}"
        )
    return result.stdout


def _print_matrix_script(var: str, nrows: int, ncols: int) -> str:
    """Generate MAGMA code to print a matrix entry-by-entry."""
    return textwrap.dedent(f"""\
        for i in [1..{nrows}] do
            for j in [1..{ncols}] do
                printf "%o\\n", {var}[i,j];
            end for;
        end for;
    """)


def _snf_magma_script(matrix: list[list[Poly]], nrows: int, ncols: int, p: int, transforms: bool) -> str:
    mat_str = _matrix_to_magma(matrix, nrows, ncols)
    transform_code = ""
    if transforms:
        transform_code = textwrap.dedent(f"""\
            printf "LEFT\\n";
            {_print_matrix_script("U", nrows, nrows)}
            printf "RIGHT\\n";
            {_print_matrix_script("V", ncols, ncols)}
        """)
    return textwrap.dedent(f"""\
        p := {p};
        R<x> := PolynomialRing(GF(p));
        M := {mat_str};
        D, U, V := SmithForm(M);
        printf "SNF\\n";
        {_print_matrix_script("D", nrows, ncols)}
        {transform_code}
        quit;
    """)


def _hnf_magma_script(matrix: list[list[Poly]], nrows: int, ncols: int, p: int, transform: bool) -> str:
    mat_str = _matrix_to_magma(matrix, nrows, ncols)
    transform_code = ""
    if transform:
        transform_code = textwrap.dedent(f"""\
            printf "LEFT\\n";
            {_print_matrix_script("U", nrows, nrows)}
        """)
    return textwrap.dedent(f"""\
        p := {p};
        R<x> := PolynomialRing(GF(p));
        M := {mat_str};
        H, U := HermiteForm(M);
        printf "HNF\\n";
        {_print_matrix_script("H", nrows, ncols)}
        {transform_code}
        quit;
    """)


def _parse_magma_output_blocks(stdout: str, p: int, shapes: dict[str, tuple[int, int]]) -> dict[str, list[list[Poly]]]:
    """Parse MAGMA output into labelled matrix blocks."""
    blocks: dict[str, list[str]] = {}
    current = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped in shapes:
            current = stripped
            blocks[current] = []
        elif current is not None and stripped:
            blocks[current].append(stripped)
    result = {}
    for label, (nr, nc) in shapes.items():
        if label in blocks:
            lines = blocks[label]
            entries = []
            for ln in lines:
                poly = _parse_poly(ln)
                poly = [c % p for c in poly]
                while poly and poly[-1] == 0:
                    poly.pop()
                entries.append(poly)
            mat = []
            for i in range(nr):
                row = [entries[i * nc + j] if (i * nc + j) < len(entries) else [] for j in range(nc)]
                mat.append(row)
            result[label] = mat
    return result


class MagmaPolyBackend(PolyBackend):
    """Uses MAGMA CLI subprocess for polynomial matrix operations over F_p[x]."""

    def compute_snf(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[Poly]]:
        script = _snf_magma_script(matrix, nrows, ncols, p, transforms=False)
        stdout = _run_magma(script)
        blocks = _parse_magma_output_blocks(stdout, p, {"SNF": (nrows, ncols)})
        snf = blocks["SNF"]
        inv = [snf[i][i] for i in range(min(nrows, ncols)) if snf[i][i]]
        return snf, inv

    def compute_snf_with_transforms(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[Poly], list[list[Poly]], list[list[Poly]]]:
        script = _snf_magma_script(matrix, nrows, ncols, p, transforms=True)
        stdout = _run_magma(script)
        blocks = _parse_magma_output_blocks(stdout, p, {
            "SNF": (nrows, ncols), "LEFT": (nrows, nrows), "RIGHT": (ncols, ncols)
        })
        snf = blocks["SNF"]
        inv = [snf[i][i] for i in range(min(nrows, ncols)) if snf[i][i]]
        return snf, inv, blocks["LEFT"], blocks["RIGHT"]

    def compute_hnf(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]]]:
        script = _hnf_magma_script(matrix, nrows, ncols, p, transform=False)
        stdout = _run_magma(script)
        blocks = _parse_magma_output_blocks(stdout, p, {"HNF": (nrows, ncols)})
        return (blocks["HNF"],)

    def compute_hnf_with_transform(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[list[Poly]]]:
        script = _hnf_magma_script(matrix, nrows, ncols, p, transform=True)
        stdout = _run_magma(script)
        blocks = _parse_magma_output_blocks(stdout, p, {"HNF": (nrows, ncols), "LEFT": (nrows, nrows)})
        return blocks["HNF"], blocks["LEFT"]

    def compute_elementary_divisors(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> list[Poly]:
        snf, inv = self.compute_snf(matrix, nrows, ncols, p)
        return inv
