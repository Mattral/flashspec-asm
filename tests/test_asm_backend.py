"""Tests of the compiled library (libflashspec_asm.so) through the bindings.

test_correctness.py tests the reference implementation only; this file runs
the shipped code: the dispatcher, the AVX2 kernel, the C baseline, and the
CPUID/XGETBV feature detection.

Needs only numpy + pytest, so it can also run against an installed wheel
(copy it outside the repo and run pytest there; it then imports the
installed package). In a source checkout it imports the repo package, which
needs `make` first. If the library is missing the module is skipped, unless
FLASHSPEC_ASM_REQUIRE_LIB=1 (set in CI), which turns that into a failure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

if importlib.util.find_spec("flashspec_asm_kernel") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import flashspec_asm_kernel as asm
except ImportError as exc:
    if os.environ.get("FLASHSPEC_ASM_REQUIRE_LIB") == "1":
        raise
    pytest.skip(f"libflashspec_asm.so not available: {exc}", allow_module_level=True)

ATOL = 1e-5
RTOL = 1e-5

# Tails around the 8-lane AVX2 width, plus real vocab sizes.
SIZES = [1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 64, 1000, 32000, 128256]


def _reference(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    r = np.maximum(p.astype(np.float64) - q.astype(np.float64), 0.0)
    return (r / max(r.sum(), 1e-9)).astype(np.float32)


def _pmfs(V: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(V)).astype(np.float32)
    q = rng.dirichlet(np.ones(V)).astype(np.float32)
    return p, q


def _cpuinfo_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


IMPLS = [
    pytest.param(asm.residual_dist, id="dispatch"),
    pytest.param(asm.residual_dist_c_baseline, id="c_baseline"),
]


def test_abi_version():
    assert asm.abi_version() == 1


@pytest.mark.parametrize("flag,fn", [
    ("avx2", asm.cpu_supports_avx2),
    ("avx512f", asm.cpu_supports_avx512f),
])
def test_cpu_feature_detection_matches_kernel(flag, fn):
    """CPUID + XGETBV must agree with what the Linux kernel reports.

    /proc/cpuinfo lists a flag only if the CPU has it AND the kernel enabled
    its register state in XCR0, the same two conditions cpuid.c checks.
    """
    flags = _cpuinfo_flags()
    if not flags:
        pytest.skip("/proc/cpuinfo flags not available")
    assert fn() == (flag in flags)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("V", SIZES)
def test_matches_numpy_reference(impl, V):
    p, q = _pmfs(V, seed=V)
    out = impl(p, q)
    assert out.dtype == np.float32 and out.shape == (V,)
    np.testing.assert_allclose(out, _reference(p, q), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("V", [1, 9, 32000])
def test_p_equals_q_hits_denom_guard(impl, V):
    p, _ = _pmfs(V, seed=1)
    out = impl(p, p.copy())
    assert np.all(out == 0.0)


@pytest.mark.parametrize("V", [17, 4096])
def test_dispatch_matches_c_baseline(V):
    p, q = _pmfs(V, seed=3)
    np.testing.assert_allclose(
        asm.residual_dist(p, q), asm.residual_dist_c_baseline(p, q),
        atol=ATOL, rtol=RTOL,
    )


def test_out_buffer_is_written_in_place():
    p, q = _pmfs(1000, seed=4)
    out = np.full(1000, np.nan, dtype=np.float32)
    ret = asm.residual_dist(p, q, out)
    assert ret is out
    np.testing.assert_allclose(out, _reference(p, q), atol=ATOL, rtol=RTOL)
