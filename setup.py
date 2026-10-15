"""Build hooks for flashspec-asm-kernel. Metadata lives in pyproject.toml.

The package is pure Python plus one shared library loaded with ctypes. It is
not a CPython extension module, so it does not depend on the interpreter
version or ABI, only on the platform. Wheels are therefore tagged
``py3-none-<platform>``: one wheel serves every Python 3 on that platform.
Release wheels are built by scripts/build_wheel.sh, which runs this inside a
manylinux2014 container and then `auditwheel repair`s the result to
``py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64``.
"""

import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.dist import Distribution

try:  # setuptools >= 70.1 ships bdist_wheel itself
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # older setuptools: provided by the `wheel` package
    from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parent
LIB = ROOT / "flashspec_asm_kernel" / "libflashspec_asm.so"


class BinaryDistribution(Distribution):
    """Ships a platform binary: not pure, so everything goes to platlib.

    Without this, setuptools treats the package as pure Python, puts the .so
    under purelib, and `auditwheel repair` rejects the wheel.
    """

    def has_ext_modules(self):
        return True


class platform_bdist_wheel(bdist_wheel):
    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", plat

    def run(self):
        # Never produce a wheel without the library. A source build (sdist,
        # or a checkout without a prior `make`) builds it here with the
        # portable flags from the Makefile; if that fails, the install fails
        # loudly instead of installing a package that cannot load.
        if not LIB.is_file():
            try:
                subprocess.run(["make", "-C", str(ROOT), "all"], check=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                raise SystemExit(
                    "flashspec-asm-kernel: building libflashspec_asm.so failed "
                    f"({exc}). A source build needs x86-64 Linux, make, nasm "
                    "and a C compiler. Prefer the binary wheel from PyPI."
                ) from exc
        super().run()


setup(
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": platform_bdist_wheel},
)
