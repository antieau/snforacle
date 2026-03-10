"""MAGMA CLI backend for matrix operations over F_p."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from snforacle.backends.ff_base import FFBackend



def _check_magma() -> None:
    if not shutil.which("magma"):
        raise RuntimeError(
            "magma binary not found on PATH. Install MAGMA to use this backend."
        )


def _run_magma(script: str) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "script.magma"
        path.write_text(script)
        result = subprocess.run(
            ["magma", "-n", str(path)], capture_output=True, text=True
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"MAGMA subprocess failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr}"
        )
    return result.stdout


def _parse_block(stdout: str, labels: list[str]) -> dict[str, list[int]]:
    """Parse labeled integer blocks from MAGMA output.

    Each block starts with a label line (e.g. ``SNF``) followed by lines of
    space-separated integers.  Returns a dict mapping label → flat list of ints.
    """
    blocks: dict[str, list[int]] = {}
    current: str | None = None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped in labels:
            current = stripped
            blocks[current] = []
        elif current is not None and stripped and stripped[0].lstrip("-").isdigit():
            blocks[current].extend(int(x) for x in stripped.split())
    return blocks


def _print_matrix_script(var: str, nrows: int, ncols: int) -> str:
    """MAGMA code to print a GF(p) matrix as a flat row of integers."""
    return textwrap.dedent(f"""\
        for _i in [1..{nrows}] do
            for _j in [1..{ncols}] do
                printf " %o", Integers()!{var}[_i,_j];
            end for;
        end for;
        printf "\\n";
    """)


def _snf_script(
    matrix: list[list[int]], nrows: int, ncols: int, p: int, transforms: bool
) -> str:
    flat = ", ".join(
        str(matrix[i][j]) for i in range(nrows) for j in range(ncols)
    )
    transform_code = ""
    if transforms:
        transform_code = textwrap.dedent(f"""\
            printf "LEFT\\n";
            {_print_matrix_script("_U", nrows, nrows)}
            printf "RIGHT\\n";
            {_print_matrix_script("_V", ncols, ncols)}
        """)
    return textwrap.dedent(f"""\
        _p := {p};
        _F := GF(_p);
        _M := Matrix(_F, {nrows}, {ncols}, [{flat}]);
        _D, _U, _V := SmithForm(_M);
        printf "SNF\\n";
        {_print_matrix_script("_D", nrows, ncols)}
        {transform_code}
        quit;
    """)


def _hnf_script(
    matrix: list[list[int]], nrows: int, ncols: int, p: int, transform: bool
) -> str:
    flat = ", ".join(
        str(matrix[i][j]) for i in range(nrows) for j in range(ncols)
    )
    transform_code = ""
    if transform:
        transform_code = textwrap.dedent(f"""\
            printf "LEFT\\n";
            {_print_matrix_script("_U", nrows, nrows)}
        """)
    return textwrap.dedent(f"""\
        _p := {p};
        _F := GF(_p);
        _M := Matrix(_F, {nrows}, {ncols}, [{flat}]);
        // Augment with identity to capture the left transform.
        _Aug := HorizontalJoin(_M, IdentityMatrix(_F, {nrows}));
        _Aug := EchelonForm(_Aug);
        _H := Submatrix(_Aug, 1, 1, {nrows}, {ncols});
        _U := Submatrix(_Aug, 1, {ncols} + 1, {nrows}, {nrows});
        printf "HNF\\n";
        {_print_matrix_script("_H", nrows, ncols)}
        {transform_code}
        quit;
    """)


def _reshape(flat: list[int], nrows: int, ncols: int) -> list[list[int]]:
    return [[flat[i * ncols + j] for j in range(ncols)] for i in range(nrows)]


class MagmaFFBackend(FFBackend):
    """Uses MAGMA CLI subprocess for F_p matrix operations."""

    def __init__(self) -> None:
        _check_magma()  # fail fast if magma is not on PATH

    def compute_snf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int]:
        stdout = _run_magma(_snf_script(matrix, nrows, ncols, p, False))
        blocks = _parse_block(stdout, ["SNF"])
        snf = _reshape(blocks["SNF"], nrows, ncols)
        rank = sum(1 for i in range(min(nrows, ncols)) if snf[i][i] != 0)
        return snf, rank

    def compute_snf_with_transforms(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], int, list[list[int]], list[list[int]]]:
        stdout = _run_magma(_snf_script(matrix, nrows, ncols, p, True))
        blocks = _parse_block(stdout, ["SNF", "LEFT", "RIGHT"])
        snf = _reshape(blocks["SNF"], nrows, ncols)
        rank = sum(1 for i in range(min(nrows, ncols)) if snf[i][i] != 0)
        left = _reshape(blocks["LEFT"], nrows, nrows)
        right = _reshape(blocks["RIGHT"], ncols, ncols)
        return snf, rank, left, right

    def compute_hnf(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]]]:
        stdout = _run_magma(_hnf_script(matrix, nrows, ncols, p, False))
        blocks = _parse_block(stdout, ["HNF"])
        return (_reshape(blocks["HNF"], nrows, ncols),)

    def compute_hnf_with_transform(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> tuple[list[list[int]], list[list[int]]]:
        stdout = _run_magma(_hnf_script(matrix, nrows, ncols, p, True))
        blocks = _parse_block(stdout, ["HNF", "LEFT"])
        return _reshape(blocks["HNF"], nrows, ncols), _reshape(blocks["LEFT"], nrows, nrows)

    def compute_rank(
        self, matrix: list[list[int]], nrows: int, ncols: int, p: int
    ) -> int:
        snf, rank = self.compute_snf(matrix, nrows, ncols, p)
        return rank
