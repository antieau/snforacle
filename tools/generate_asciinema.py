#!/usr/bin/env python3
"""generate_asciinema.py — Animate the Smith Normal Form algorithm step by step.

Usage:
    python tools/generate_asciinema.py MATRIX [OUTPUT]

    MATRIX  JSON list-of-lists of integers, e.g. '[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]'
    OUTPUT  .cast output path (default: snf_demo.cast)

Play back with:
    asciinema play snf_demo.cast
    asciinema play --speed 2 snf_demo.cast
"""
from __future__ import annotations

import json
import re
import sys
from typing import Callable

# ── ANSI colour codes ─────────────────────────────────────────────────────────
R   = "\033[0m"
B   = "\033[1m"
DIM = "\033[2m"
FG_W = "\033[97m"; FG_Y = "\033[93m"; FG_C = "\033[96m"
FG_G = "\033[92m"; FG_D = "\033[37m"
BG_HDR   = "\033[44m"    # blue header
BG_PIVOT = "\033[43m"    # amber – pivot cell
BG_SRC   = "\033[100m"   # dark-grey – source row/col
BG_TGT   = "\033[46m"    # cyan – target row/col
BG_SW1   = "\033[45m"    # magenta – first swap item
BG_SW2   = "\033[104m"   # bright-blue – second swap item
CLR = "\033[2J\033[H"    # clear + cursor home

_ANSI = re.compile(r"\033\[[0-9;]*m")

def _vlen(s: str) -> int:
    """Visual length (strips ANSI escape codes)."""
    return len(_ANSI.sub("", s))

def _pad(s: str, w: int) -> str:
    return s + " " * max(0, w - _vlen(s))

# ── Splash screen ─────────────────────────────────────────────────────────────
#
# "snforacle" in Calvin S figlet font (box-drawing characters, 3 rows × 4 chars/letter):
#
#    ╔═╗ ╔╗  ╔══ ╔═╗ ╦═╗ ╔═╗ ╔═╗ ╦   ╔═╗
#    ╚═╗ ║╚╗ ╠══ ║ ║ ╠╦╝ ╠═╣ ║   ║   ╠═
#    ╚═╝ ╝  ╝╩   ╚═╝ ╩╚═ ╩ ╩ ╚═╝ ╚══ ╚═╝

_ART = [
    "   ╔═╗ ╔╗  ╔══ ╔═╗ ╦═╗ ╔═╗ ╔═╗ ╦   ╔═╗ ",
    "   ╚═╗ ║╚╗ ╠══ ║ ║ ╠╦╝ ╠═╣ ║   ║   ╠═  ",
    "   ╚═╝ ╝  ╝╩   ╚═╝ ╩╚═ ╩ ╩ ╚═╝ ╚══ ╚═╝ ",
]

def _splash_lines() -> list[str]:
    """Return ANSI-coloured splash-screen lines (no CLR prefix)."""
    art_col  = f"{B}\033[94m"   # bold bright-blue for the box-drawing art
    sep_col  = DIM
    math_col = FG_C
    key_col  = FG_Y
    dim_col  = FG_D

    return [
        "",
        f"{art_col}{_ART[0]}{R}",
        f"{art_col}{_ART[1]}{R}",
        f"{art_col}{_ART[2]}{R}",
        "",
        f"   {sep_col}{'─' * 36}{R}",
        f"   {math_col}Smith Normal Form · Hermite Normal Form · Elementary Divisors{R}",
        f"   for  ℤ  ·  F_p  ·  F_p[x]  —  pluggable multi-backend Python API",
        "",
        f"   {key_col}Backends:{R}  PARI/GP (cypari2)  ·  FLINT (python-flint)",
        f"              {dim_col}SageMath  ·  MAGMA  ·  pure Python (reference){R}",
        "",
    ]

def _splash_frame(term_w: int, term_h: int) -> str:
    lines = _splash_lines()
    # vertically centre in the terminal
    top_pad = max(0, (term_h - len(lines)) // 2)
    ls = [""] * top_pad + lines
    while len(ls) < term_h:
        ls.append("")
    return CLR + "\r\n".join(ls[:term_h])

# ── Extended GCD ──────────────────────────────────────────────────────────────

def _xgcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) with a·x + b·y = g = gcd(|a|,|b|)."""
    if b == 0:
        return abs(a), (1 if a >= 0 else -1), 0
    r0, r1, s0, s1 = a, b, 1, 0
    while r1:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
    g = abs(r0)
    x = s0 if r0 > 0 else -s0
    return g, x, (g - a * x) // b

# ── SNF event recorder ────────────────────────────────────────────────────────

def _snf_events(M0: list[list[int]]) -> list[dict]:
    """Run the SNF algorithm and return a list of operation events.

    Each event dict has:
      type   – operation name
      k      – diagonal index being processed
      desc   – human-readable description
      before – (M, U, V) snapshot before the operation
      after  – (M, U, V) snapshot after the operation
    """
    m, n = len(M0), len(M0[0]) if M0 else 0
    M = [list(r) for r in M0]
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]
    evs: list[dict] = []

    def snap():
        return [list(r) for r in M], [list(r) for r in U], [list(r) for r in V]

    def emit(**kw) -> int:
        evs.append({"before": snap(), **kw})
        return len(evs) - 1

    def close(i: int):
        evs[i]["after"] = snap()

    emit(type="start", k=0, desc="Goal: find diagonal D and invertible U, V  with  D = U·M·V")
    close(0)

    for k in range(min(m, n)):
        # outer: repeat until divisibility condition is satisfied
        while True:
            # inner: clear row k and column k
            while True:
                best = min(
                    ((i, j) for i in range(k, m) for j in range(k, n) if M[i][j]),
                    key=lambda p: abs(M[p[0]][p[1]]),
                    default=None,
                )
                if best is None:
                    break

                pi, pj = best

                if pi != k:
                    i = emit(type="swap_rows", r1=k, r2=pi, k=k,
                             desc=f"Move smallest |{M[pi][pj]}| to pivot: swap row {k} ↔ row {pi}")
                    M[k], M[pi] = M[pi], M[k]
                    U[k], U[pi] = U[pi], U[k]
                    close(i)

                if pj != k:
                    i = emit(type="swap_cols", c1=k, c2=pj, k=k,
                             desc=f"Move smallest to pivot: swap col {k} ↔ col {pj}")
                    for row in M: row[k], row[pj] = row[pj], row[k]
                    for row in V: row[k], row[pj] = row[pj], row[k]
                    close(i)

                if M[k][k] < 0:
                    i = emit(type="negate_row", row=k, k=k,
                             desc=f"Make pivot positive: negate row {k}  (was {M[k][k]})")
                    M[k] = [-v for v in M[k]]
                    U[k] = [-v for v in U[k]]
                    close(i)

                piv = M[k][k]
                progress = False

                for t in range(k, m):       # eliminate column k
                    if t == k or not M[t][k]:
                        continue
                    a, b = piv, M[t][k]
                    if b % a == 0:
                        q = b // a
                        i = emit(type="elim_row", source=k, target=t, q=q, k=k,
                                 desc=f"Row {t} ← Row {t} − {q}·Row {k}   (zeros out M[{t}][{k}] = {b})")
                        for j in range(n): M[t][j] -= q * M[k][j]
                        for j in range(m): U[t][j] -= q * U[k][j]
                        close(i)
                    else:
                        g, x, y = _xgcd(a, b)
                        i = emit(type="gcd_rows", r1=k, r2=t, g=g, x=x, y=y, a=a, b=b, k=k,
                                 desc=f"GCD rows {k}&{t}: gcd({a},{b})={g}  →  "
                                      f"row{k} ← {x}·r{k} + ({y})·r{t}")
                        nk = [x*M[k][j] + y*M[t][j] for j in range(n)]
                        nt = [(a//g)*M[t][j] - (b//g)*M[k][j] for j in range(n)]
                        M[k], M[t] = nk, nt
                        nUk = [x*U[k][j] + y*U[t][j] for j in range(m)]
                        nUt = [(a//g)*U[t][j] - (b//g)*U[k][j] for j in range(m)]
                        U[k], U[t] = nUk, nUt
                        close(i)
                    progress = True

                piv = M[k][k]
                for t in range(k, n):       # eliminate row k
                    if t == k or not M[k][t]:
                        continue
                    a, b = piv, M[k][t]
                    if b % a == 0:
                        q = b // a
                        i = emit(type="elim_col", source=k, target=t, q=q, k=k,
                                 desc=f"Col {t} ← Col {t} − {q}·Col {k}   (zeros out M[{k}][{t}] = {b})")
                        for r in range(m): M[r][t] -= q * M[r][k]
                        for r in range(n): V[r][t] -= q * V[r][k]
                        close(i)
                    else:
                        g, x, y = _xgcd(a, b)
                        i = emit(type="gcd_cols", c1=k, c2=t, g=g, x=x, y=y, a=a, b=b, k=k,
                                 desc=f"GCD cols {k}&{t}: gcd({a},{b})={g}  →  "
                                      f"col{k} ← {x}·c{k} + ({y})·c{t}")
                        nk = [x*M[r][k] + y*M[r][t] for r in range(m)]
                        nt = [(a//g)*M[r][t] - (b//g)*M[r][k] for r in range(m)]
                        for r in range(m): M[r][k], M[r][t] = nk[r], nt[r]
                        nVk = [x*V[r][k] + y*V[r][t] for r in range(n)]
                        nVt = [(a//g)*V[r][t] - (b//g)*V[r][k] for r in range(n)]
                        for r in range(n): V[r][k], V[r][t] = nVk[r], nVt[r]
                        close(i)
                    progress = True

                if not progress:
                    break

            if not M[k][k]:
                break

            # divisibility check: M[k][k] must divide every entry in the remaining submatrix
            viol = next(
                ((i, j) for i in range(k+1, m) for j in range(k+1, n)
                 if M[i][j] % M[k][k]),
                None,
            )
            if viol is None:
                break
            vi, vj = viol
            i = emit(type="div_fix", k=k, vi=vi, vj=vj,
                     desc=f"Divisibility: M[{k}][{k}]={M[k][k]} does not divide "
                          f"M[{vi}][{vj}]={M[vi][vj]}  →  col {k} ← col {k} + col {vj}")
            for r in range(m): M[r][k] += M[r][vj]
            for r in range(n): V[r][k] += V[r][vj]
            close(i)

    i = emit(type="done", k=min(m, n),
             desc="Diagonal entries are the invariant factors; D = U·M·V")
    close(i)
    return evs

# ── Matrix drawing ────────────────────────────────────────────────────────────

def _col_w(matrices: list[list[list[int]]], floor: int = 2) -> int:
    """Compute the column display width needed to show all values (including sign)."""
    vals = [abs(v) for M in matrices for row in M for v in row]
    return max(floor, len(str(max(vals, default=0))) + 1)


def _draw(
    M: list[list[int]],
    w: int,
    fn: Callable[[int, int], str] | None = None,
    label: str = "",
) -> list[str]:
    """Render matrix M as a boxed grid.  fn(i,j) returns an optional ANSI style string."""
    if not M or not M[0]:
        lbl = [f"  {DIM}{label}{R}"] if label else []
        return lbl + ["  (empty)"]
    rows, cols = len(M), len(M[0])
    bar = "─" * (w + 2)
    top = "┌" + "┬".join(bar for _ in range(cols)) + "┐"
    bot = "└" + "┴".join(bar for _ in range(cols)) + "┘"
    out = []
    if label:
        out.append(f"  {DIM}{label}{R}")
    out.append("  " + top)
    for r, row in enumerate(M):
        cells = []
        for c, v in enumerate(row):
            st = (fn(r, c) if fn else "") or ""
            cells.append(f" {st}{str(v).rjust(w)}{R} ")
        out.append("  │" + "│".join(cells) + "│")
    out.append("  " + bot)
    return out


def _beside(panels: list[list[str]], gap: int = 4) -> list[str]:
    """Lay panels out side by side, padding each to its maximum line width."""
    if not panels:
        return []
    h = max(len(p) for p in panels)
    ws = [max((_vlen(l) for l in p), default=0) for p in panels]
    sp = " " * gap
    return [
        sp.join(
            _pad(panels[pi][r] if r < len(panels[pi]) else "", ws[pi])
            for pi in range(len(panels))
        )
        for r in range(h)
    ]

# ── Terminal size computation ─────────────────────────────────────────────────

def _compute_term_size(m: int, n: int, wM: int, wU: int, wV: int) -> tuple[int, int]:
    """Return (term_w, term_h) tightly fitted to the matrix and column widths."""
    # Visual width of each panel (max of grid line width and label width)
    def panel_vis_w(num_cols: int, col_w: int, label: str) -> int:
        grid_w = num_cols * (col_w + 3) + 3   # "  " + border + cells
        return max(grid_w, len(label))

    pM = panel_vis_w(n, wM, "  M  (working matrix)")
    pU = panel_vis_w(m, wU, "  U  (left transform)")
    pV = panel_vis_w(n, wV, "  V  (right transform)")
    beside_w = pM + 5 + pU + 5 + pV   # gap=5 between panels

    # Legend line visual width ~95; give a small buffer
    term_w = max(beside_w, 96) + 2

    # Frame line count:
    # 1 header + 1 blank + 1 title + 1 desc + 1 blank
    # + beside_height (= max(m,n)+3 for _draw with label)
    # + 1 blank + 1 inv_factors + 1 blank + 1 legend = besides + 9
    beside_h = max(m, n) + 3
    term_h = beside_h + 9 + 2   # +2 buffer rows

    # Splash screen is 13 lines; ensure it fits too
    term_h = max(term_h, 15)
    return term_w, term_h

# ── Frame renderer ────────────────────────────────────────────────────────────

def _frame(
    M: list[list[int]],
    U: list[list[int]],
    V: list[list[int]],
    *,
    title: str,
    desc: str,
    k: int,
    wM: int,
    wU: int,
    wV: int,
    fn: Callable[[int, int], str] | None,
    term_w: int,
    term_h: int,
    badge: str = "",
) -> str:
    """Render a complete terminal frame as an ANSI string."""
    ls: list[str] = []

    # header bar
    hdr = f"{BG_HDR}{FG_W}{B}  Smith Normal Form — step by step  {R}"
    ls.append(_pad(hdr + ("  " + badge if badge else ""), term_w))
    ls.append("")

    # step title + description (always two lines so matrices don't shift)
    ls.append(f"  {FG_Y}{B}{title}{R}")
    ls.append(f"  {FG_D}{desc}{R}" if desc else "")
    ls.append("")

    # done-diagonal styler (green tint for completed positions)
    def _done(i: int, j: int) -> str:
        return FG_G if (i == j < k) else ""

    def _style_M(i: int, j: int) -> str:
        return (fn(i, j) if fn else "") or _done(i, j)

    pM = _draw(M, wM, _style_M, "M  (working matrix)")
    pU = _draw(U, wU, _done,    "U  (left transform)")
    pV = _draw(V, wV, _done,    "V  (right transform)")
    for ln in _beside([pM, pU, pV], gap=5):
        ls.append(ln)
    ls.append("")

    # invariant factors accumulated so far (nonzero diagonal entries only)
    r = min(k, len(M), len(M[0]) if M else 0)
    nz = [M[i][i] for i in range(r) if M[i][i] != 0]
    if nz:
        ls.append(f"  {DIM}Invariant factors ({len(nz)} so far):  {FG_G}{B}{', '.join(str(f) for f in nz)}{R}")

    # colour legend (only when highlighting is active)
    if fn is not None:
        ls += [
            "",
            f"  {BG_PIVOT}   {R} pivot  "
            f"  {BG_SRC}   {R} source row/col  "
            f"  {BG_TGT}   {R} target row/col  "
            f"  {BG_SW1}   {R}/{BG_SW2}   {R} swap pair  "
            f"  {FG_G}■{R} done diagonal",
        ]

    while len(ls) < term_h:
        ls.append("")
    return CLR + "\r\n".join(ls[:term_h])

# ── Per-event highlight builders ─────────────────────────────────────────────

def _highlight(ev: dict) -> tuple[Callable[[int, int], str] | None, str]:
    """Return (cell_style_fn, short_title) for the PRE ("about to") frame."""
    t  = ev["type"]
    k  = ev.get("k", 0)

    if t == "start":
        return None, "Initial matrix"
    if t == "done":
        return None, "Smith Normal Form complete ✓"

    if t == "swap_rows":
        r1, r2 = ev["r1"], ev["r2"]
        return (
            lambda i, j, _r1=r1, _r2=r2: BG_SW1 if i == _r1 else (BG_SW2 if i == _r2 else "")
        ), f"Swap row {r1} ↔ row {r2}"

    if t == "swap_cols":
        c1, c2 = ev["c1"], ev["c2"]
        return (
            lambda i, j, _c1=c1, _c2=c2: BG_SW1 if j == _c1 else (BG_SW2 if j == _c2 else "")
        ), f"Swap col {c1} ↔ col {c2}"

    if t == "negate_row":
        row = ev["row"]
        return (
            lambda i, j, _row=row: BG_SRC if i == _row else ""
        ), f"Negate row {row}"

    if t == "elim_row":
        src, tgt, q = ev["source"], ev["target"], ev["q"]
        return (
            lambda i, j, _src=src, _tgt=tgt, _k=k: (
                BG_PIVOT if (i == _src and j == _k) else
                BG_SRC   if (i == _src) else
                BG_TGT   if (i == _tgt) else ""
            )
        ), f"Row {tgt} ← Row {tgt} − {q}·Row {src}"

    if t == "elim_col":
        src, tgt, q = ev["source"], ev["target"], ev["q"]
        return (
            lambda i, j, _src=src, _tgt=tgt, _k=k: (
                BG_PIVOT if (i == _k and j == _src) else
                BG_SRC   if (j == _src) else
                BG_TGT   if (j == _tgt) else ""
            )
        ), f"Col {tgt} ← Col {tgt} − {q}·Col {src}"

    if t == "gcd_rows":
        r1, r2 = ev["r1"], ev["r2"]
        return (
            lambda i, j, _r1=r1, _r2=r2: BG_SW1 if i == _r1 else (BG_SW2 if i == _r2 else "")
        ), f"GCD step on rows {r1} & {r2}"

    if t == "gcd_cols":
        c1, c2 = ev["c1"], ev["c2"]
        return (
            lambda i, j, _c1=c1, _c2=c2: BG_SW1 if j == _c1 else (BG_SW2 if j == _c2 else "")
        ), f"GCD step on cols {c1} & {c2}"

    if t == "div_fix":
        vi, vj = ev["vi"], ev["vj"]
        return (
            lambda i, j, _k=k, _vi=vi, _vj=vj: (
                BG_PIVOT if (i == _k and j == _k) else
                BG_TGT   if (i == _vi and j == _vj) else
                BG_SRC   if (j == _vj) else ""
            )
        ), f"Divisibility fix: col {k} ← col {k} + col {vj}"

    return None, t

# ── Cast writer ───────────────────────────────────────────────────────────────

DUR_SPLASH = 3.0  # splash screen hold time
DUR_INTRO  = 2.5  # "start" frame hold time
DUR_PRE    = 2.2  # "about to" (highlighted) frame hold time
DUR_POST   = 1.4  # "result" frame hold time
DUR_FINAL  = 4.0  # "done" frame hold time


def _write_cast(evs: list[dict], path: str) -> None:
    # Compute fixed column widths from all intermediate states
    all_M = [s[0] for ev in evs for s in (ev["before"], ev["after"])]
    all_U = [s[1] for ev in evs for s in (ev["before"], ev["after"])]
    all_V = [s[2] for ev in evs for s in (ev["before"], ev["after"])]
    wM, wU, wV = _col_w(all_M), _col_w(all_U), _col_w(all_V)

    # Matrix dimensions (from first snapshot)
    M0 = evs[0]["before"][0]
    m, n = len(M0), len(M0[0]) if M0 else 0
    term_w, term_h = _compute_term_size(m, n, wM, wU, wV)

    cast: list[list] = []
    t = 0.0
    total = len(evs)

    # ── Splash screen ──────────────────────────────────────────────────────────
    cast.append([round(t, 3), "o", _splash_frame(term_w, term_h)])
    t += DUR_SPLASH

    # ── Algorithm frames ───────────────────────────────────────────────────────
    frame_kw = dict(wM=wM, wU=wU, wV=wV, term_w=term_w, term_h=term_h)

    for si, ev in enumerate(evs):
        etype = ev["type"]
        k     = ev.get("k", 0)
        desc  = ev.get("desc", "")
        fn, short_title = _highlight(ev)
        M_pre, U_pre, V_pre   = ev["before"]
        M_post, U_post, V_post = ev["after"]

        if etype == "start":
            frame = _frame(
                M_pre, U_pre, V_pre,
                title=f"Step 0/{total-1}  ·  {short_title}",
                desc=desc, k=k, fn=fn,
                badge=f"{BG_HDR}{FG_C} START {R}",
                **frame_kw,
            )
            cast.append([round(t, 3), "o", frame])
            t += DUR_INTRO

        elif etype == "done":
            frame = _frame(
                M_post, U_post, V_post,
                title=f"Step {si}/{total-1}  ·  {short_title}",
                desc=desc, k=k, fn=None,
                badge=f"\033[42m{FG_W} DONE {R}",
                **frame_kw,
            )
            cast.append([round(t, 3), "o", frame])
            t += DUR_FINAL

        else:
            # PRE frame: highlight what is about to happen
            pre = _frame(
                M_pre, U_pre, V_pre,
                title=f"Step {si}/{total-1}  ·  ▶ {short_title}",
                desc=desc, k=k, fn=fn,
                badge=f"{BG_PIVOT} NEXT {R}",
                **frame_kw,
            )
            cast.append([round(t, 3), "o", pre])
            t += DUR_PRE

            # POST frame: show the result without highlights
            post = _frame(
                M_post, U_post, V_post,
                title=f"Step {si}/{total-1}  ·  ✓ {short_title}",
                desc="", k=k, fn=None,
                badge=f"\033[42m{FG_W} OK {R}",
                **frame_kw,
            )
            cast.append([round(t, 3), "o", post])
            t += DUR_POST

    # trailing no-op to keep the last frame visible
    cast.append([round(t + 0.1, 3), "o", ""])

    header = {
        "version":   2,
        "width":     term_w,
        "height":    term_h,
        "timestamp": 1743500000,
        "title":     "Smith Normal Form — step by step",
    }
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for ev in cast:
            f.write(json.dumps(ev) + "\n")

    n_steps = sum(1 for ev in evs if ev["type"] not in ("start", "done"))
    print(f"Wrote {len(cast)} frames  ({n_steps} algorithm steps)  →  {path}")
    print(f"Terminal size: {term_w}×{term_h}")
    print(f"Duration: {t:.1f} s  (use --speed 2 for faster playback)")
    print(f"Play:  asciinema play {path}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    try:
        M = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(M, list) or not M or not isinstance(M[0], list):
        print("Error: expected a JSON list-of-lists, e.g. '[[1,2],[3,4]]'", file=sys.stderr)
        sys.exit(1)
    if any(len(row) != len(M[0]) for row in M):
        print("Error: rows must all have the same length", file=sys.stderr)
        sys.exit(1)

    output = sys.argv[2] if len(sys.argv) > 2 else "snf_demo.cast"
    evs = _snf_events(M)
    _write_cast(evs, output)


if __name__ == "__main__":
    main()
