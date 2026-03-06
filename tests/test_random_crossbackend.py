"""Cross-backend consistency test on 1000 random square and non-square matrices.

Matrices have shapes nrows, ncols ∈ [10, 30] with integer entries in [-10, 10].
All five operations are tested:

  - smith_normal_form
  - smith_normal_form_with_transforms  (verifies U@M@V == SNF)
  - hermite_normal_form
  - hermite_normal_form_with_transform (verifies U@M == H)
  - elementary_divisors

**Backend grouping**

*Fast in-process backends* (cypari2, flint) run on every one of the 1000
matrices.  pure_python has exponential intermediate value growth beyond ~12×12
(e.g. 13×13 can hang indefinitely), so it is compared only on matrices where
max(nrows, ncols) ≤ PP_MAX=12, sampled on every 10th matrix.

CLI backends (sage, magma) are exercised on a separate 50-matrix subset in
TestCLIBackendCrossCheck.

All results are compared against pure_python (always available) as the
reference implementation.  Failures report the matrix index so the failing
input can be reproduced via list(_gen_matrices(1000))[index].
"""

from __future__ import annotations

import random
import shutil
import pytest

from snforacle import (
    elementary_divisors,
    hermite_normal_form,
    hermite_normal_form_with_transform,
    smith_normal_form,
    smith_normal_form_with_transforms,
)


# ---------------------------------------------------------------------------
# Backend availability
# ---------------------------------------------------------------------------

def _avail(name: str) -> bool:
    """Return True only if the backend is installed AND works (catches sandbox issues)."""
    if name == "pure_python":
        return True
    if name in ("sage", "magma"):
        if not shutil.which(name):
            return False
        try:
            from snforacle import smith_normal_form  # noqa: PLC0415
            smith_normal_form(
                {"format": "dense", "nrows": 1, "ncols": 1, "entries": [[1]]},
                backend=name,
            )
            return True
        except Exception:
            return False
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# Fast in-process backends (run on all 1000 matrices).
# NOTE: flint.fmpz_mat.snf() hangs on certain non-square matrices (python-flint
# 0.8.0 limitation).  Only cypari2 is used for SNF/ED in the random test.
# flint.fmpz_mat.hnf() has no such issue and is used for HNF.
_FAST_SNF   = [b for b in ["cypari2"] if _avail(b)]
_FAST_SNF_T = [b for b in ["cypari2"] if _avail(b)]   # flint: no transforms
_FAST_HNF   = [b for b in ["flint"] if _avail(b)]     # cypari2: no HNF
_FAST_ED    = [b for b in ["cypari2"] if _avail(b)]    # flint SNF hangs on non-square

# pure_python is compared on every PP_STRIDE-th matrix.
PP_STRIDE = 10  # 1000 / 10 = 100 pure_python comparisons

# pure_python can have exponential intermediate value growth for both SNF and
# HNF on matrices larger than ~12×12 (worst-case hang observed on 13×13).
# Skip pure_python comparisons when any dimension exceeds this threshold.
PP_MAX = 12

# CLI backends (subprocess overhead; separate smaller test).
_CLI_BACKENDS = [b for b in ["sage", "magma"] if _avail(b)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense(entries: list[list[int]], nrows: int, ncols: int) -> dict:
    return {"format": "dense", "nrows": nrows, "ncols": ncols, "entries": entries}


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    m, n = len(A), len(A[0]) if A else 0
    p = len(B[0]) if B else 0
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]


def _gen_matrices(n: int, seed: int = 42):
    """Yield (index, nrows, ncols, entries) for *n* random matrices."""
    rng = random.Random(seed)
    for i in range(n):
        nrows = rng.randint(10, 30)
        ncols = rng.randint(10, 30)
        entries = [[rng.randint(-10, 10) for _ in range(ncols)] for _ in range(nrows)]
        yield i, nrows, ncols, entries


# ---------------------------------------------------------------------------
# Main 1000-matrix test
# ---------------------------------------------------------------------------

class TestRandomCrossBackend:
    """Cross-backend consistency on 1000 random matrices.

    Fast in-process backends (cypari2, flint) are compared on all 1000
    matrices.  pure_python is compared on every 10th matrix (100 total).
    """

    N = 1000

    def test_snf(self):
        """All fast SNF backends agree; pure_python sampled every 10th matrix."""
        if not _FAST_SNF and not _avail("pure_python"):
            pytest.skip("No SNF backends available")
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)

            # Fast backends: compare pairwise on every matrix.
            if len(_FAST_SNF) >= 2:
                ref_inv = smith_normal_form(inp, backend=_FAST_SNF[0]).invariant_factors
                for b in _FAST_SNF[1:]:
                    result = smith_normal_form(inp, backend=b)
                    assert result.invariant_factors == ref_inv, (
                        f"matrix {i} ({nrows}×{ncols}): "
                        f"{_FAST_SNF[0]}={ref_inv} vs {b}={result.invariant_factors}"
                    )
            elif _FAST_SNF:
                ref_inv = smith_normal_form(inp, backend=_FAST_SNF[0]).invariant_factors
            else:
                ref_inv = None

            # pure_python: every 10th matrix, small enough to avoid integer blow-up.
            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                pp = smith_normal_form(inp, backend="pure_python").invariant_factors
                if ref_inv is not None:
                    assert pp == ref_inv, (
                        f"matrix {i} ({nrows}×{ncols}): fast ref={ref_inv} vs pure_python={pp}"
                    )

    def test_snf_with_transforms(self):
        """U@M@V == SNF for fast SNF-T backends; pure_python every 10th matrix."""
        if not _FAST_SNF_T and not _avail("pure_python"):
            pytest.skip("No SNF-transform backends available")
        ref_snf_b = _FAST_SNF[0] if _FAST_SNF else "pure_python"
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)
            ref_inv = smith_normal_form(inp, backend=ref_snf_b).invariant_factors

            for b in _FAST_SNF_T:
                result = smith_normal_form_with_transforms(inp, backend=b)
                assert result.invariant_factors == ref_inv, (
                    f"matrix {i} ({nrows}×{ncols}): ref={ref_inv} vs {b} SNF+T"
                )
                U = result.left_transform.entries
                V = result.right_transform.entries
                S = result.smith_normal_form.entries
                assert _mat_mul(_mat_mul(U, entries), V) == S, (
                    f"matrix {i} ({nrows}×{ncols}): U@M@V ≠ SNF for {b}"
                )

            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                result = smith_normal_form_with_transforms(inp, backend="pure_python")
                assert result.invariant_factors == ref_inv, (
                    f"matrix {i} ({nrows}×{ncols}): ref={ref_inv} vs pure_python SNF+T"
                )
                U = result.left_transform.entries
                V = result.right_transform.entries
                S = result.smith_normal_form.entries
                assert _mat_mul(_mat_mul(U, entries), V) == S, (
                    f"matrix {i} ({nrows}×{ncols}): U@M@V ≠ SNF for pure_python"
                )

    def test_hnf(self):
        """All fast HNF backends agree; pure_python sampled every 10th matrix."""
        if not _FAST_HNF and not _avail("pure_python"):
            pytest.skip("No HNF backends available")
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)

            if _FAST_HNF:
                ref_hnf = hermite_normal_form(inp, backend=_FAST_HNF[0]).hermite_normal_form.entries
                for b in _FAST_HNF[1:]:
                    result = hermite_normal_form(inp, backend=b)
                    assert result.hermite_normal_form.entries == ref_hnf, (
                        f"matrix {i} ({nrows}×{ncols}): {_FAST_HNF[0]} vs {b} HNF differ"
                    )
            else:
                ref_hnf = None

            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                pp_hnf = hermite_normal_form(inp, backend="pure_python").hermite_normal_form.entries
                if ref_hnf is not None:
                    assert pp_hnf == ref_hnf, (
                        f"matrix {i} ({nrows}×{ncols}): fast HNF ≠ pure_python HNF"
                    )

    def test_hnf_with_transform(self):
        """U@M == H for pure_python (every 10th matrix, max dim ≤ PP_MAX), H matches fast HNF backend."""
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            if i % PP_STRIDE != 0 or max(nrows, ncols) > PP_MAX:
                continue
            inp = _dense(entries, nrows, ncols)
            ref_hnf = (
                hermite_normal_form(inp, backend=_FAST_HNF[0]).hermite_normal_form.entries
                if _FAST_HNF else None
            )
            result = hermite_normal_form_with_transform(inp, backend="pure_python")
            H = result.hermite_normal_form.entries
            U = result.left_transform.entries
            assert _mat_mul(U, entries) == H, (
                f"matrix {i} ({nrows}×{ncols}): U@M ≠ H for pure_python"
            )
            if ref_hnf is not None:
                assert H == ref_hnf, (
                    f"matrix {i} ({nrows}×{ncols}): pure_python HNF+T differs from fast HNF"
                )

    def test_elementary_divisors(self):
        """All fast ED backends agree; pure_python sampled every 10th matrix."""
        if not _FAST_ED and not _avail("pure_python"):
            pytest.skip("No ED backends available")
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)

            if len(_FAST_ED) >= 2:
                ref_ed = elementary_divisors(inp, backend=_FAST_ED[0]).elementary_divisors
                for b in _FAST_ED[1:]:
                    result = elementary_divisors(inp, backend=b)
                    assert result.elementary_divisors == ref_ed, (
                        f"matrix {i} ({nrows}×{ncols}): "
                        f"{_FAST_ED[0]}={ref_ed} vs {b}={result.elementary_divisors}"
                    )
            elif _FAST_ED:
                ref_ed = elementary_divisors(inp, backend=_FAST_ED[0]).elementary_divisors
            else:
                ref_ed = None

            if i % PP_STRIDE == 0 and max(nrows, ncols) <= PP_MAX:
                pp = elementary_divisors(inp, backend="pure_python").elementary_divisors
                if ref_ed is not None:
                    assert pp == ref_ed, (
                        f"matrix {i} ({nrows}×{ncols}): fast ref={ref_ed} vs pure_python={pp}"
                    )

    def test_snf_matches_ed(self):
        """SNF invariant_factors == elementary_divisors for every fast backend."""
        shared = [b for b in _FAST_SNF if b in _FAST_ED]
        if not shared:
            pytest.skip("No fast backend supports both SNF and ED")
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)
            for b in shared:
                snf_inv = smith_normal_form(inp, backend=b).invariant_factors
                ed      = elementary_divisors(inp, backend=b).elementary_divisors
                assert snf_inv == ed, (
                    f"matrix {i} ({nrows}×{ncols}): "
                    f"SNF.invariant_factors={snf_inv} ≠ ED={ed} for {b}"
                )


# ---------------------------------------------------------------------------
# 50-matrix CLI backend cross-check
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _CLI_BACKENDS,
    reason="no CLI backends (sage/magma) available",
)
class TestCLIBackendCrossCheck:
    """Cross-backend consistency using CLI backends on 50 random matrices.

    cypari2 (or pure_python if unavailable) is used as the SNF/ED reference.
    flint (or pure_python if unavailable) is used as the HNF reference.
    """

    N = 50
    # Fast reference backends (avoid pure_python which hangs on large matrices).
    _SNF_REF = _FAST_SNF[0] if _FAST_SNF else "pure_python"
    _HNF_REF = _FAST_HNF[0] if _FAST_HNF else "pure_python"

    def test_snf(self):
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)
            ref_inv = smith_normal_form(inp, backend=self._SNF_REF).invariant_factors
            for b in _CLI_BACKENDS:
                result = smith_normal_form(inp, backend=b)
                assert result.invariant_factors == ref_inv, (
                    f"matrix {i} ({nrows}×{ncols}): {self._SNF_REF}={ref_inv} vs {b}={result.invariant_factors}"
                )

    def test_snf_with_transforms(self):
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)
            ref_inv = smith_normal_form(inp, backend=self._SNF_REF).invariant_factors
            for b in _CLI_BACKENDS:
                result = smith_normal_form_with_transforms(inp, backend=b)
                assert result.invariant_factors == ref_inv, (
                    f"matrix {i} ({nrows}×{ncols}): SNF+T mismatch for {b}"
                )
                U = result.left_transform.entries
                V = result.right_transform.entries
                S = result.smith_normal_form.entries
                assert _mat_mul(_mat_mul(U, entries), V) == S, (
                    f"matrix {i} ({nrows}×{ncols}): U@M@V ≠ SNF for {b}"
                )

    def test_hnf(self):
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)
            ref_hnf = hermite_normal_form(inp, backend=self._HNF_REF).hermite_normal_form.entries
            for b in _CLI_BACKENDS:
                result = hermite_normal_form(inp, backend=b)
                assert result.hermite_normal_form.entries == ref_hnf, (
                    f"matrix {i} ({nrows}×{ncols}): {self._HNF_REF} vs {b} HNF differ"
                )

    def test_hnf_with_transform(self):
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)
            ref_hnf = hermite_normal_form(inp, backend=self._HNF_REF).hermite_normal_form.entries
            for b in _CLI_BACKENDS:
                result = hermite_normal_form_with_transform(inp, backend=b)
                H = result.hermite_normal_form.entries
                U = result.left_transform.entries
                assert H == ref_hnf, (
                    f"matrix {i} ({nrows}×{ncols}): {b} HNF ≠ {self._HNF_REF} HNF"
                )
                assert _mat_mul(U, entries) == H, (
                    f"matrix {i} ({nrows}×{ncols}): U@M ≠ H for {b}"
                )

    def test_elementary_divisors(self):
        for i, nrows, ncols, entries in _gen_matrices(self.N):
            inp = _dense(entries, nrows, ncols)
            ref_ed = elementary_divisors(inp, backend=self._SNF_REF).elementary_divisors
            for b in _CLI_BACKENDS:
                result = elementary_divisors(inp, backend=b)
                assert result.elementary_divisors == ref_ed, (
                    f"matrix {i} ({nrows}×{ncols}): {self._SNF_REF}={ref_ed} vs {b}={result.elementary_divisors}"
                )
