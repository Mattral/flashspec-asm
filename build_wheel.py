#!/usr/bin/env python3
"""
Custom wheel builder for flashspec-asm-kernel.

cibuildwheel rejects wheels that bundle a .so as package data (ctypes pattern)
rather than as a compiled Python extension module. This script builds the wheel
correctly by:
  1. Running make to compile libflashspec_asm.so
  2. Copying the .so into the package directory
  3. Building the wheel with the correct platform tag

Run from repo root:
    python build_wheel.py
"""
import os, sys, shutil, subprocess, sysconfig, platform
from pathlib import Path

ROOT = Path(__file__).parent
PKG  = ROOT / "flashspec_asm_kernel"
DIST = ROOT / "dist"

def get_platform_tag():
    """Return the manylinux platform tag for the current machine."""
    machine = platform.machine()  # x86_64
    # Use python -c to get the exact tag setuptools would use
    result = subprocess.run(
        [sys.executable, "-c",
         "from wheel.bdist_wheel import get_platform; print(get_platform(None))"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip().replace("-", "_").replace(".", "_")
    return f"linux_{machine}"

def build():
    DIST.mkdir(exist_ok=True)

    # Step 1: compile the .so
    print("Building libflashspec_asm.so ...")
    subprocess.run(["make", "clean"], cwd=ROOT, check=True)
    subprocess.run(["make"],          cwd=ROOT, check=True)

    so = ROOT / "libflashspec_asm.so"
    assert so.exists(), "make did not produce libflashspec_asm.so"

    # Step 2: copy into package
    dest = PKG / "libflashspec_asm.so"
    shutil.copy2(so, dest)
    print(f"Copied .so → {dest}")

    # Step 3: build the wheel (setuptools produces py3-none-any)
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
        cwd=ROOT, check=True
    )

    # Step 4: retag to the correct platform (mimics what auditwheel repair does)
    wheels = list(DIST.glob("*.whl"))
    assert wheels, "No wheel produced"
    whl = wheels[-1]

    platform_tag = get_platform_tag()
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    new_name = whl.name.replace(
        "py3-none-any",
        f"{py_tag}-{py_tag}-{platform_tag}"
    )
    new_path = DIST / new_name
    whl.rename(new_path)
    print(f"\nWheel ready: {new_path}")

if __name__ == "__main__":
    build()
