"""MAGMA CLI backend for polynomial matrices over F_p[x]."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

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


def _trim(coeffs: list[int]) -> Poly:
    c = list(coeffs)
    while c and c[-1] == 0:
        c.pop()
    return c


def _parse_poly_line(line: str) -> Poly:
    """Parse a coefficient-sequence line produced by MAGMA's Eltseq output.

    Each line has the format ``<length> <c0> <c1> ... <cd>`` where *length* is
    the number of coefficients (0 for the zero polynomial) and the *ci* are
    integers.  This avoids the fragility of parsing MAGMA's human-readable
    polynomial expressions.
    """
    parts = line.split()
    if not parts:
        return []
    n = int(parts[0])
    if n == 0:
        return []
    coeffs = [int(parts[i + 1]) for i in range(n)]
    return _trim(coeffs)


def _inv_mod_p(c: int, p: int) -> int:
    return pow(c, p - 2, p)


def _make_monic(poly: Poly, p: int) -> Poly:
    """Return the monic associate of *poly* (divide by leading coefficient)."""
    if not poly:
        return []
    lc_inv = _inv_mod_p(poly[-1], p)
    return [c * lc_inv % p for c in poly]


def _parse_magma_matrix(lines: list[str], nrows: int, ncols: int) -> list[list[Poly]]:
    """Parse nrows*ncols coefficient-sequence lines into a polynomial matrix."""
    entries = [_parse_poly_line(ln) for ln in lines]
    mat = []
    for i in range(nrows):
        row = [entries[i * ncols + j] if (i * ncols + j) < len(entries) else [] for j in range(ncols)]
        mat.append(row)
    return mat


def _run_magma(script: str) -> str:
    _check_magma()
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "script.magma"
        script_path.write_text(script)
        result = subprocess.run(
            ["magma", "-n", str(script_path)],
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"MAGMA subprocess failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr}"
        )
    return result.stdout


def _print_matrix_script(var: str, nrows: int, ncols: int) -> str:
    """Generate MAGMA code to print a matrix as Eltseq coefficient lines.

    Each entry is printed as ``<length> <c0> <c1> ... <cd>\\n``, where
    *length* is the number of coefficients (0 for the zero polynomial) and
    *ci* are integers.  This format is unambiguous and trivially parsed in
    Python without regular expressions.
    """
    return textwrap.dedent(f"""\
        for _i in [1..{nrows}] do
            for _j in [1..{ncols}] do
                _seq := Eltseq({var}[_i,_j]);
                printf "%o", #_seq;
                for _k in [1..#_seq] do
                    printf " %o", Integers()!_seq[_k];
                end for;
                printf "\\n";
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


def _parse_magma_output_blocks(stdout: str, shapes: dict[str, tuple[int, int]]) -> dict[str, list[list[Poly]]]:
    """Parse MAGMA output into labelled matrix blocks.

    Each block starts with a label line (e.g. ``SNF``) followed by
    ``nrows * ncols`` coefficient-sequence lines, one per matrix entry.
    """
    blocks: dict[str, list[str]] = {}
    current = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped in shapes:
            current = stripped
            blocks[current] = []
        elif current is not None and stripped and stripped[0].isdigit():
            blocks[current].append(stripped)
    result = {}
    for label, (nr, nc) in shapes.items():
        if label in blocks:
            result[label] = _parse_magma_matrix(blocks[label], nr, nc)
    return result


class MagmaPolyBackend(PolyBackend):
    """Uses MAGMA CLI subprocess for polynomial matrix operations over F_p[x]."""

    def __init__(self) -> None:
        _check_magma()  # fail fast if magma is not on PATH

    def compute_snf(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[Poly]]:
        script = _snf_magma_script(matrix, nrows, ncols, p, transforms=False)
        stdout = _run_magma(script)
        blocks = _parse_magma_output_blocks(stdout, {"SNF": (nrows, ncols)})
        snf = blocks["SNF"]
        # Normalize diagonal to monic (MAGMA may return non-monic invariant factors)
        for i in range(min(nrows, ncols)):
            if snf[i][i]:
                snf[i][i] = _make_monic(snf[i][i], p)
        inv = [snf[i][i] for i in range(min(nrows, ncols)) if snf[i][i]]
        return snf, inv

    def compute_snf_with_transforms(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[Poly], list[list[Poly]], list[list[Poly]]]:
        script = _snf_magma_script(matrix, nrows, ncols, p, transforms=True)
        stdout = _run_magma(script)
        blocks = _parse_magma_output_blocks(stdout, {
            "SNF": (nrows, ncols), "LEFT": (nrows, nrows), "RIGHT": (ncols, ncols)
        })
        snf = blocks["SNF"]
        left = blocks["LEFT"]
        # Normalize diagonal to monic; scale corresponding row of U to maintain U·M·V = D
        for i in range(min(nrows, ncols)):
            if snf[i][i]:
                lc = snf[i][i][-1]
                if lc != 1:
                    lc_inv = _inv_mod_p(lc, p)
                    snf[i][i] = _make_monic(snf[i][i], p)
                    for j in range(nrows):
                        left[i][j] = [c * lc_inv % p for c in left[i][j]]
                        left[i][j] = _trim(left[i][j])
        inv = [snf[i][i] for i in range(min(nrows, ncols)) if snf[i][i]]
        return snf, inv, left, blocks["RIGHT"]

    def compute_hnf(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]]]:
        script = _hnf_magma_script(matrix, nrows, ncols, p, transform=False)
        stdout = _run_magma(script)
        blocks = _parse_magma_output_blocks(stdout, {"HNF": (nrows, ncols)})
        return (blocks["HNF"],)

    def compute_hnf_with_transform(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[Poly]], list[list[Poly]]]:
        script = _hnf_magma_script(matrix, nrows, ncols, p, transform=True)
        stdout = _run_magma(script)
        blocks = _parse_magma_output_blocks(stdout, {"HNF": (nrows, ncols), "LEFT": (nrows, nrows)})
        return blocks["HNF"], blocks["LEFT"]

    def compute_elementary_divisors(
        self, matrix: list[list[Poly]], nrows: int, ncols: int, p: int
    ) -> list[Poly]:
        snf, inv = self.compute_snf(matrix, nrows, ncols, p)
        return inv
