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


# ---------------------------------------------------------------------------
# Argument validation (0.1.2). The C side takes raw pointers plus V = len(p),
# so each of these used to be an out-of-bounds access or a silently ignored
# output buffer.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("Vq", [0, 1, 63, 65])
def test_length_mismatch_raises(impl, Vq):
    """A shorter q was read past its end; a longer one was silently cut."""
    p, _ = _pmfs(64, seed=5)
    q = np.full(Vq, 1.0 / max(Vq, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="same length"):
        impl(p, q)


@pytest.mark.parametrize("impl", IMPLS)
def test_dtype_mismatch_raises(impl):
    p, q = _pmfs(64, seed=6)
    with pytest.raises(ValueError, match="same dtype"):
        impl(p, q.astype(np.float64))
    with pytest.raises(ValueError, match="same dtype"):
        impl(p.astype(np.float16), q)


@pytest.mark.parametrize("impl", IMPLS)
def test_matching_non_float32_inputs_still_accepted(impl):
    """p and q are only read, so matching float64 inputs are converted."""
    p, q = _pmfs(100, seed=7)
    out = impl(p.astype(np.float64), q.astype(np.float64))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, _reference(p, q), atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("impl", IMPLS)
def test_non_1d_raises(impl):
    p, q = _pmfs(64, seed=8)
    with pytest.raises(ValueError, match="1-D"):
        impl(p.reshape(8, 8), q.reshape(8, 8))
    with pytest.raises(ValueError, match="1-D"):
        impl(np.float32(0.5), np.float32(0.5))


def _bad_outs(V: int):
    base = np.zeros(2 * V, dtype=np.float32)
    readonly = np.zeros(V, dtype=np.float32)
    readonly.flags.writeable = False
    return [
        pytest.param(np.zeros(V, dtype=np.float64), "dtype float32", id="float64"),
        pytest.param(np.zeros(V, dtype=np.float16), "dtype float32", id="float16"),
        pytest.param(np.zeros(V, dtype=">f4"), "dtype float32", id="big-endian-f4"),
        pytest.param(base[::2], "C-contiguous", id="strided"),
        pytest.param(np.zeros(V - 1, dtype=np.float32), r"shape \(64,\)", id="too-short"),
        pytest.param(np.zeros(V + 1, dtype=np.float32), r"shape \(64,\)", id="too-long"),
        pytest.param(np.zeros((1, V), dtype=np.float32), r"shape \(64,\)", id="2-D"),
        pytest.param(readonly, "writeable", id="read-only"),
        pytest.param([0.0] * V, "numpy.ndarray", id="list"),
    ]


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("out,match", _bad_outs(64))
def test_bad_out_buffer_raises_and_is_untouched(impl, out, match):
    """Never write the result to a hidden copy (or past the buffer's end)."""
    p, q = _pmfs(64, seed=9)
    before = np.array(out, copy=True)
    with pytest.raises(ValueError, match=match):
        impl(p, q, out)
    np.testing.assert_array_equal(np.asarray(out), before)


@pytest.mark.parametrize("impl", IMPLS)
def test_valid_out_buffer_written_in_place(impl):
    p, q = _pmfs(257, seed=10)
    out = np.full(257, np.nan, dtype=np.float32)
    assert impl(p, q, out) is out
    np.testing.assert_allclose(out, _reference(p, q), atol=ATOL, rtol=RTOL)
