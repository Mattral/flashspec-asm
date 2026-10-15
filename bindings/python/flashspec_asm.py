"""Legacy location of the FlashSpec-ASM ctypes bindings.

The bindings live in ``flashspec_asm_kernel/flashspec_asm.py`` (the package
that ships on PyPI). This module re-exports them so code that puts
``bindings/python`` on sys.path (e.g. bench/bench_harness.py) keeps working
with one copy of the loader instead of two diverging ones.

Like the package, importing this raises ImportError if libflashspec_asm.so
is missing or cannot be loaded. Build it with `make` from the repo root.
"""

import pathlib
import sys

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flashspec_asm_kernel.flashspec_asm import (  # noqa: E402,F401
    abi_version,
    cpu_supports_avx2,
    cpu_supports_avx512f,
    residual_dist,
    residual_dist_c_baseline,
)
