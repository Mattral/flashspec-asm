#!/usr/bin/env python3
"""Smoke-test an INSTALLED flashspec-asm-kernel, from outside the source tree.

Run it with the interpreter of a fresh venv into which only the wheel (and
its dependencies) was installed, from a directory that is not the repo, e.g.:

    python -m venv /tmp/v && /tmp/v/bin/pip install wheelhouse/*.whl
    cd /tmp && /tmp/v/bin/python -I /path/to/scripts/smoke_test_wheel.py

Fails (non-zero exit) unless:
  * the package and its shared library both come from site-packages, not from
    a source checkout (and the library is the one actually mapped into this
    process, per /proc/self/maps);
  * the CPU supports AVX2, i.e. residual_dist dispatches to the ASM kernel
    (pass --allow-no-avx2 to only warn, for runners without AVX2);
  * residual_dist (and the C baseline) agree with a NumPy reference.
"""

from __future__ import annotations

import argparse
import sys
import sysconfig
from pathlib import Path

import numpy as np


def reference(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    r = np.maximum(p.astype(np.float64) - q.astype(np.float64), 0.0)
    return (r / max(r.sum(), 1e-9)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-no-avx2", action="store_true")
    args = ap.parse_args()

    import flashspec_asm_kernel as asm
    from flashspec_asm_kernel import flashspec_asm as binding

    site_dirs = {Path(sysconfig.get_paths()[k]).resolve()
                 for k in ("purelib", "platlib")}
    pkg_file = Path(asm.__file__).resolve()
    lib_file = Path(binding._LIB_PATH).resolve()
    print(f"python          : {sys.version.split()[0]} ({sys.executable})")
    print(f"version         : {asm.__version__}")
    print(f"package         : {pkg_file}")
    print(f"library         : {lib_file}")

    def in_site(path: Path) -> bool:
        return any(d in path.parents for d in site_dirs)

    assert in_site(pkg_file), f"package not imported from site-packages {site_dirs}"
    assert in_site(lib_file), f"library not loaded from site-packages {site_dirs}"

    maps = Path("/proc/self/maps").read_text()
    mapped = {line.split()[-1] for line in maps.splitlines()
              if line.rstrip().endswith("libflashspec_asm.so")}
    print(f"mapped          : {sorted(mapped)}")
    assert mapped == {str(lib_file)}, "a different libflashspec_asm.so is mapped"

    abi = asm.abi_version()
    avx2 = asm.cpu_supports_avx2()
    print(f"abi_version()   : {abi}")
    print(f"avx2 / avx512f  : {avx2} / {asm.cpu_supports_avx512f()}")
    assert abi == 1, abi
    if not avx2:
        msg = "CPU lacks AVX2: residual_dist uses the C fallback, not the ASM kernel"
        if not args.allow_no_avx2:
            raise AssertionError(msg)
        print(f"WARNING: {msg}")
    print(f"backend         : {'ASM (residual_dist_avx2)' if avx2 else 'C fallback'}")

    rng = np.random.default_rng(0)
    worst = 0.0
    for V in (1, 7, 8, 9, 17, 1000, 32000, 128256):
        p = rng.dirichlet(np.ones(V)).astype(np.float32)
        q = rng.dirichlet(np.ones(V)).astype(np.float32)
        ref = reference(p, q)
        for fn in (asm.residual_dist, asm.residual_dist_c_baseline):
            out = fn(p, q)
            assert out.dtype == np.float32 and out.shape == (V,)
            np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5,
                                       err_msg=f"{fn.__name__} V={V}")
            worst = max(worst, float(np.max(np.abs(out - ref))))

    p = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float32)
    q = np.array([0.2, 0.3, 0.1, 0.4], dtype=np.float32)
    out = asm.residual_dist(p, q)
    print(f"residual_dist({p.tolist()}, {q.tolist()})")
    print(f"                = {out.tolist()}")
    print(f"numpy reference = {reference(p, q).tolist()}")
    print(f"max |asm - numpy| over V in 1..128256: {worst:.3g}")
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
