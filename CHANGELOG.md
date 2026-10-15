# Changelog

All notable changes to `flashspec-asm-kernel` are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.1.1] - unreleased

0.1.0 on PyPI is a source distribution only. Installing it never built the
shared library, so the installed package could not run anything. 0.1.1 fixes
the packaging. The C ABI (`FLASHSPEC_ASM_ABI_VERSION = 1`), the function
signatures and the AVX2 kernel are unchanged.

### Fixed
- **Binary wheel.** Release wheels are built in the
  `quay.io/pypa/manylinux2014_x86_64` container (`scripts/build_wheel.sh`)
  and run through `auditwheel repair`, tagged
  `py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64`. The library is
  loaded with ctypes, not imported as a CPython extension, so one wheel
  serves every Python 3. This replaces `build_wheel.py`, which renamed the
  file to a `linux_x86_64` tag that PyPI rejects.
- **SIGILL in the non-AVX2 fallback.** The C files were compiled with
  `-march=native`. On an AVX/AVX-512 build host, that put AVX instructions
  into `residual_dist_c`, the fallback for CPUs *without* AVX2. `make` now
  builds the C code for baseline x86-64 (`-march=x86-64 -mtune=generic`).
  `make bench-native` keeps the `-march=native` build for benchmarking.
  `scripts/check_fallback_isa.py` disassembles the shipped library and
  fails CI if any non-kernel entry point contains VEX/EVEX instructions.
- **Missing library now fails at import.** If `libflashspec_asm.so` is
  missing or cannot be loaded, `import flashspec_asm_kernel` raises
  `ImportError`. In 0.1.0 the import succeeded and the first call raised
  `OSError`. Cause: the loader wrapped `ctypes.util.find_library()`'s `None`
  in `Path("")`, i.e. `Path(".")`, which always exists, and then tried to
  dlopen `.`. The library is now loaded eagerly at import.
- **XCR0 read in CPU feature detection** (`src/cpuid.c`). XGETBV is now read
  as EDX:EAX (`"=a"`, `"=d"`) and combined. On x86-64 the `"=A"` constraint
  does not mean the EDX:EAX pair; it captured only RAX, dropping XCR0's high
  half. Results on current CPUs are unchanged, because all tested bits are
  in the low half.
- `cpu_supports_avx2()` / `cpu_supports_avx512f()` check that CPUID leaf 7
  exists before reading it. `cpu_supports_avx512f()` also requires XCR0
  bit 7 (Hi16_ZMM state), as the Intel SDM specifies.
- Source builds (sdist, `pip install .`) run `make`, so they produce a
  working install or fail loudly, never a package without a library. The
  sdist ships the sources and never a prebuilt `.so`.

### Added
- `tests/test_asm_backend.py`: tests the compiled library (dispatcher, AVX2
  kernel, C baseline, CPUID detection compared with `/proc/cpuinfo`) against
  NumPy. Before this, the test suite exercised only the reference
  implementation.
- `tests/test_packaging.py`: an install without the library, or with an
  unloadable one, raises `ImportError` at import.
- CI on every pull request: build the wheel in the manylinux container, run
  `auditwheel show`, the ISA check, and install tests in clean venvs
  (Python 3.9/3.12/3.13, plus glibc 2.17). These tests assert that the
  package and library load from site-packages and that the ASM kernel is
  the dispatched backend.
- `.github/workflows/release.yml`: a manual-only, disabled publish skeleton
  gated on the wheel tests.

### Changed
- `bindings/python/flashspec_asm.py` re-exports `flashspec_asm_kernel`
  instead of carrying a second copy of the loader.

## [0.1.0]

Initial release (source distribution only; see 0.1.1).
