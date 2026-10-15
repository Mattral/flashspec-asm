"""Loader behaviour when the package is installed without a usable library.

flashspec-asm-kernel 0.1.0 shipped only an sdist, so installs had no .so.
Its loader turned that into an import that succeeded and a first call that
raised OSError (it tried to dlopen "."), which `except ImportError`
fallbacks do not catch. These tests pin the fixed contract: a missing or
broken library raises ImportError at import time.

Each case copies the package's Python files (no .so) to a temp dir and
imports it in a fresh, isolated interpreter, so neither the repo checkout
nor an installed copy can satisfy the import.
"""

from __future__ import annotations

import ctypes.util
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parent.parent / "flashspec_asm_kernel"
LIB_NAME = "libflashspec_asm.so"

pytestmark = pytest.mark.skipif(
    ctypes.util.find_library("flashspec_asm") is not None,
    reason="a libflashspec_asm is installed system-wide; the loader would find it",
)

PROBE = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    try:
        import flashspec_asm_kernel
    except ImportError as exc:
        print("IMPORT_ERROR:", exc)
        sys.exit(0)
    except BaseException as exc:
        print("WRONG_EXCEPTION:", type(exc).__name__, exc)
        sys.exit(2)
    print("IMPORTED_OK:", flashspec_asm_kernel.__file__)
    sys.exit(1)
""")


def _copy_package_without_lib(dest: Path) -> Path:
    pkg = dest / "flashspec_asm_kernel"
    pkg.mkdir()
    for src in PKG.glob("*.py"):
        shutil.copy2(src, pkg / src.name)
    assert not (pkg / LIB_NAME).exists()
    return pkg


def _import_in_subprocess(site: Path) -> subprocess.CompletedProcess:
    # -I: ignore PYTHONPATH, user site and the cwd; cwd is the temp dir anyway.
    return subprocess.run(
        [sys.executable, "-I", "-c", PROBE, str(site)],
        cwd=site, capture_output=True, text=True, timeout=120,
    )


def test_missing_library_raises_importerror_at_import(tmp_path):
    _copy_package_without_lib(tmp_path)
    proc = _import_in_subprocess(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "IMPORT_ERROR:" in proc.stdout
    assert LIB_NAME in proc.stdout and "not found" in proc.stdout


def test_unloadable_library_raises_importerror_not_oserror(tmp_path):
    pkg = _copy_package_without_lib(tmp_path)
    (pkg / LIB_NAME).write_bytes(b"this is not an ELF shared object")
    proc = _import_in_subprocess(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "IMPORT_ERROR:" in proc.stdout
    assert "could not be loaded" in proc.stdout

