"""FlashSpec-ASM Python bindings (Phase 4).

Wraps libflashspec_asm.so via ctypes. Provides a drop-in replacement for
the pure-NumPy residual distribution used in flashspec's CPU fallback path.

Usage (after Phase 4 ships):
    from flashspec_asm import residual_dist, cpu_supports_avx2, abi_version

Integration into flashspec:
    try:
        import flashspec_asm as _asm
        _ASM_AVAILABLE = _asm.cpu_supports_avx2()
        if _asm.abi_version() != _EXPECTED_ABI:
            import warnings
            warnings.warn(
                "flashspec-asm ABI mismatch — falling back to CPU path. "
                f"Expected ABI {_EXPECTED_ABI}, got {_asm.abi_version()}."
            )
            _ASM_AVAILABLE = False
    except ImportError:
        _ASM_AVAILABLE = False

Status: Phase 4 stub. Functions raise NotImplementedError until the shared
object is built and the ctypes bindings are wired up.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import pathlib
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Expected ABI version — must match FLASHSPEC_ASM_ABI_VERSION in the header.
# Bump here whenever the C signature changes (and bump the header too).
# ---------------------------------------------------------------------------
_EXPECTED_ABI_VERSION = 1

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _find_library() -> Optional[pathlib.Path]:
    """Search for libflashspec_asm.so in standard locations."""
    candidates = [
        # Installed alongside this Python file (wheel layout)
        pathlib.Path(__file__).parent / "libflashspec_asm.so",
        # Repo root (development build via `make`)
        pathlib.Path(__file__).parent.parent / "libflashspec_asm.so",
        # Legacy bindings/python location
        pathlib.Path(__file__).parent.parent / "bindings" / "python" / "libflashspec_asm.so",
        # System library path
        pathlib.Path(ctypes.util.find_library("flashspec_asm") or ""),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


_lib: Optional[ctypes.CDLL] = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    path = _find_library()
    if path is None:
        raise ImportError(
            "libflashspec_asm.so not found. "
            "Build it with `make` from the repo root, or install the wheel."
        )

    _lib = ctypes.CDLL(str(path))
    _setup_signatures(_lib)
    return _lib


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Bind C types to ctypes function objects."""
    # int abi_version(void)
    lib.abi_version.restype  = ctypes.c_int
    lib.abi_version.argtypes = []

    # int cpu_supports_avx2(void)
    lib.cpu_supports_avx2.restype  = ctypes.c_int
    lib.cpu_supports_avx2.argtypes = []

    # int cpu_supports_avx512f(void)
    lib.cpu_supports_avx512f.restype  = ctypes.c_int
    lib.cpu_supports_avx512f.argtypes = []

    # void residual_dist(const float* p, const float* q, float* out, int64_t V)
    lib.residual_dist.restype  = None
    lib.residual_dist.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int64,
    ]

    # void residual_dist_c(const float* p, const float* q, float* out, int64_t V)
    lib.residual_dist_c.restype  = None
    lib.residual_dist_c.argtypes = lib.residual_dist.argtypes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def abi_version() -> int:
    """Return the ABI version compiled into the shared object."""
    return int(_load().abi_version())


def cpu_supports_avx2() -> bool:
    """Return True if the runtime CPU supports AVX2."""
    return bool(_load().cpu_supports_avx2())


def cpu_supports_avx512f() -> bool:
    """Return True if the runtime CPU supports AVX-512F (stretch goal)."""
    return bool(_load().cpu_supports_avx512f())


def residual_dist(
    p: np.ndarray,
    q: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute residual distribution via the dispatch wrapper.

    Uses the AVX2 kernel if the CPU supports it, otherwise falls back to
    the C baseline. Matches the output of ``reference_op.residual_dist_single``
    within ATOL=1e-5 on non-NaN inputs (see ADR 0001 for NaN contract).

    Parameters
    ----------
    p   : float32 array, shape (V,) — target probabilities
    q   : float32 array, shape (V,) — draft probabilities
    out : optional pre-allocated float32 output buffer, shape (V,)

    Returns
    -------
    out : float32 array, shape (V,) — residual pmf
    """
    p = np.ascontiguousarray(p, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError(f"p and q must be 1-D arrays of equal shape; got {p.shape}, {q.shape}")

    V = p.shape[0]
    if out is None:
        out = np.empty(V, dtype=np.float32)
    else:
        out = np.ascontiguousarray(out, dtype=np.float32)

    lib = _load()
    lib.residual_dist(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(V),
    )
    return out


def residual_dist_c_baseline(
    p: np.ndarray,
    q: np.ndarray,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute residual distribution using the naive C baseline only.

    Used in the benchmark harness to isolate the C baseline from the
    dispatch wrapper. Never routes to the AVX2 kernel.
    """
    p = np.ascontiguousarray(p, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    V = p.shape[0]
    if out is None:
        out = np.empty(V, dtype=np.float32)

    lib = _load()
    lib.residual_dist_c(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(V),
    )
    return out
