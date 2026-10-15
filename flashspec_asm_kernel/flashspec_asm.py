"""FlashSpec-ASM Python bindings (Phase 4).

Wraps libflashspec_asm.so via ctypes. Provides a drop-in replacement for
the pure-NumPy residual distribution used in flashspec's CPU fallback path.

Usage:
    from flashspec_asm_kernel import residual_dist, cpu_supports_avx2, abi_version

The shared library is loaded when this module is imported. If it is missing
or cannot be loaded, the import raises ImportError, so callers only need
``except ImportError`` to fall back:

    try:
        import flashspec_asm_kernel as _asm
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
"""

from __future__ import annotations

import ctypes
import ctypes.util
import pathlib
from typing import Iterator, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Expected ABI version — must match FLASHSPEC_ASM_ABI_VERSION in the header.
# Bump here whenever the C signature changes (and bump the header too).
# ---------------------------------------------------------------------------
_EXPECTED_ABI_VERSION = 1

_LIB_NAME = "libflashspec_asm.so"

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def _candidates() -> Iterator[str]:
    """Library locations to try, in order.

    1. Next to this file: the wheel layout, and where `make` copies the
       library in a source checkout.
    2. The dynamic linker's search path, by name (only consulted if 1 is
       missing or fails to load, since find_library spawns subprocesses).
       find_library returns a soname such as "libflashspec_asm.so", not a
       path, so it is handed to dlopen as is, and None (nothing found) is
       skipped. 0.1.0 wrapped it in Path(... or ""), i.e. Path("."), which
       always "exists", so a missing library was "found" as ".".
    """
    local = pathlib.Path(__file__).resolve().parent / _LIB_NAME
    if local.is_file():
        yield str(local)
    system = ctypes.util.find_library("flashspec_asm")
    if system:
        yield system


def _load() -> tuple[ctypes.CDLL, str]:
    errors = []
    for path in _candidates():
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        try:
            _setup_signatures(lib)
        except AttributeError as exc:
            # A library by that name, but not ours (or a stale build).
            errors.append(f"{path}: missing symbol: {exc}")
            continue
        return lib, path
    if not errors:
        raise ImportError(
            f"{_LIB_NAME} not found next to {__file__} or on the dynamic "
            "linker path. Install a binary wheel of flashspec-asm-kernel, "
            "or build it with `make` from a source checkout."
        )
    raise ImportError(
        f"{_LIB_NAME} could not be loaded:\n  " + "\n  ".join(errors)
    )


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


# Load eagerly: a missing or broken library must fail the import (ImportError)
# rather than the first call (OSError), so `except ImportError` fallbacks work.
_lib, _LIB_PATH = _load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def abi_version() -> int:
    """Return the ABI version compiled into the shared object."""
    return int(_lib.abi_version())


def cpu_supports_avx2() -> bool:
    """Return True if the runtime CPU supports AVX2."""
    return bool(_lib.cpu_supports_avx2())


def cpu_supports_avx512f() -> bool:
    """Return True if the runtime CPU supports AVX-512F (stretch goal)."""
    return bool(_lib.cpu_supports_avx512f())


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

    _lib.residual_dist(
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

    _lib.residual_dist_c(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(V),
    )
    return out
