# FlashSpec-ASM — Roadmap

## Release model

**Independent repo, optional PyPI dependency.** FlashSpec-ASM ships as its own package
(`flashspec-asm-kernel` on PyPI) and integrates into `flashspec` as a soft optional
dependency — not a source merge. This follows the pattern of Flash Attention, xformers,
and bitsandbytes.

```toml
# flashspec/pyproject.toml (to be added once v1 ships)
[project.optional-dependencies]
asm = ["flashspec-asm-kernel>=0.1.0"]
```

Users install `pip install flashspec` as before; `pip install flashspec[asm]` opts into
the accelerated CPU path. Existing users see zero change.

## Integration architecture

```python
# flashspec/backends/verify.py  (to be added in flashspec repo post v1)
try:
    import flashspec_asm_kernel as _asm
    _ASM_AVAILABLE = _asm.cpu_supports_avx2()
except ImportError:
    _ASM_AVAILABLE = False

def residual_dist(p, q, out, vocab_size, use_asm=True):
    if use_asm and _ASM_AVAILABLE:
        return _asm.residual_dist(p, q, out, vocab_size)
    return _cpu_fallback(p, q, out, vocab_size)  # existing path, unchanged
```

## ABI versioning

Because `flashspec` will depend on `flashspec-asm-kernel`'s C ABI across independent
release cycles, a safety check prevents silent ABI mismatch corruption:

```python
# checked at import time in flashspec
EXPECTED_ASM_ABI = 1
if _asm.abi_version() != EXPECTED_ASM_ABI:
    warnings.warn("flashspec-asm-kernel ABI mismatch — falling back to CPU path")
    _ASM_AVAILABLE = False
```

Mismatch → silent fallback to Triton/CPU path, never a silent wrong result.
ADR to be written once integration PR is opened against flashspec.

---

## Phase status

| Phase | Description | Status |
|---|---|---|
| 0 | Profile verification step; identify hot-path op | ✅ Done — `residual_dist` at 92.5% avg |
| 1 | NumPy/PyTorch reference + property-based test suite | ✅ Done — 34 tests, all green |
| 2 | Naive C baseline (`-O3 -march=native`), benchmarked | ✅ Done — numbers in `bench/results/phase2_*.txt` |
| 3 | Hand-written AVX2 NASM kernel | ✅ Done — bug found & fixed; 13/13 correctness cases |
| 4 | C shim + Python bindings (`ctypes`/`cffi`) | ✅ Done — dispatch wired, ABI version check |
| 5 | Cycle-level benchmark (C vs ASM) | ✅ Done — 4.85× over C at V=32k; see `bench/results/phase5_*.txt` |
| 6 | ADRs, README, writeup | ✅ Done — 4 ADRs, writeup ready for publication |
| 7 | Integration PR into `flashspec` as optional dep | ⬜ Post-v1 |
| 8 | Stretch: CPUID dispatch for AVX-512 or ARM NEON | ⬜ Post-v1 |

## Phase 0 finding

Profiled all sub-operations of the verification pipeline across 5 realistic
parameter scenarios (B=1–8, γ=4–8, V=32000–128256):

| Operation | Avg % of pipeline |
|---|---|
| `residual_distribution` | **92.5%** |
| `first_rejection` | 1.5% |
| `acceptance_criterion` | 0.7% |
| `gather_logprobs` | 0.7% |

**Target op: `residual_dist_single`** — the per-sequence residual distribution:

```
diff    = p - q               (vsubps)
clamped = max(0, diff)        (vmaxps)
denom   = sum(clamped)        (horizontal reduce)
out     = clamped / denom     (vdivps or rcp + vmulps)
```

This is dominated by the vocab-size loop (V=32000–128256), making it an ideal
candidate for AVX2 vectorisation with 8-wide float32 ymm registers.

See `bench/results/phase0_*.txt` for raw numbers.

## Known constraints

- v1 targets Linux x86-64 only (ELF, SysV AMD64 ABI)
- No Windows calling convention support in v1
- AVX2 baseline only; CPUID scalar fallback for safety
- AVX-512 and ARM NEON are stretch goals gated behind full v1 release
