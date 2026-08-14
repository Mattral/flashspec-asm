"""flashspec-asm-kernel: AVX2 assembly kernel for speculative decoding residual distribution.

Optional accelerated backend for FlashSpec. Activates automatically when
installed alongside flashspec on a CPU with AVX2 support.

Install
-------
    pip install flashspec[asm]       # via flashspec optional dependency
    pip install flashspec-asm-kernel  # standalone

Usage
-----
    import flashspec_asm_kernel as asm

    if asm.cpu_supports_avx2() and asm.abi_version() == 1:
        out = asm.residual_dist(p, q)   # p, q: float32 numpy arrays shape (V,)
"""

from .flashspec_asm import (  # noqa: F401
    abi_version,
    cpu_supports_avx2,
    cpu_supports_avx512f,
    residual_dist,
    residual_dist_c_baseline,
)

__version__ = "0.1.0"
__all__ = [
    "abi_version",
    "cpu_supports_avx2",
    "cpu_supports_avx512f",
    "residual_dist",
    "residual_dist_c_baseline",
]
