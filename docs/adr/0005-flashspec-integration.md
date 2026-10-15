# ADR 0005 — FlashSpec Integration Architecture

**Status:** Accepted  
**Phase:** 7 (Integration PR into `flashspec`)

---

## Context

Phase 7 requires wiring `flashspec-asm-kernel` into `flashspec` as an optional
accelerated backend for the CPU residual distribution path. The integration must:

1. Be transparent to existing users (`pip install flashspec` behaviour unchanged)
2. Activate automatically when `flashspec[asm]` is installed and AVX2 is present
3. Never produce wrong results — ABI mismatch must fall back, not corrupt
4. Touch the minimum possible surface area of `flashspec` source

## Integration point identified

Reading `flashspec/sampling/rejection.py`, the exact three lines to replace are
in `_sample_residual()`:

```python
# Lines 181-183 of rejection.py (flashspec v0.1.4)
residual = torch.clamp(p_at_rej - q_at_rej, min=0.0)
denom = residual.sum(dim=-1, keepdim=True).clamp(min=_MIN_PROB)
residual = residual / denom
```

This is the only change to `flashspec` source. Everything else — the Triton
acceptance kernel, first_rejection logic, multinomial sampling, bandit
orchestration — is untouched.

## Decision

**Single file change:** `flashspec/sampling/rejection.py` only.

**Optional import at module level** (not per-call):

```python
_EXPECTED_ASM_ABI: int = 1
_ASM_AVAILABLE: bool = False
_asm_residual_dist = None

try:
    import flashspec_asm_kernel as _asm_kernel
    if _asm_kernel.abi_version() != _EXPECTED_ASM_ABI:
        warnings.warn("ABI mismatch — falling back", RuntimeWarning)
    elif _asm_kernel.cpu_supports_avx2():
        _ASM_AVAILABLE = True
        _asm_residual_dist = _asm_kernel.residual_dist
except (ImportError, OSError):
    _ASM_AVAILABLE = False
    _asm_residual_dist = None
```

Catch **both** exceptions. From 0.1.1 on, a missing or unloadable library
raises `ImportError` at import. But 0.1.0 (sdist only, no library) imported
fine and raised `OSError` from the first call, i.e. from `abi_version()`
inside this `try`. With `except ImportError` alone, that `OSError` escaped
and made `import flashspec.sampling.rejection` crash.

**Runtime routing conditions** — ASM path used only when ALL of:
- `_ASM_AVAILABLE` is True (kernel installed, ABI matches, AVX2 present)
- `p_at_rej.device.type == "cpu"` (never intercept GPU tensors)
- `p_at_rej.dtype == torch.float32 and q_at_rej.dtype == torch.float32`
  (the kernel computes in float32 and writes into `residual` through
  `.numpy()`; float64/bf16/fp16 take the pure-Python path)
- `p_at_rej.is_contiguous()` (kernel requires contiguous float32 memory)

The float32 guard is required, not an optimisation. With a float64 tensor,
kernel 0.1.1 wrote the result into a float32 *copy* of the output buffer
and left `residual` as uninitialised `torch.empty` memory, a silent wrong
sample. bf16/fp16 tensors fail in `.numpy()`. From 0.1.2 the kernel raises
`ValueError` on a non-float32, non-contiguous or wrong-length `out`, and
on `p`/`q` of different lengths or dtypes, so a missing guard is a crash
rather than garbage. The guard keeps those calls from happening at all.

Otherwise falls back to the existing pure-Python path, unchanged.

**Per-batch-element loop** — the C ABI takes a single `(V,)` slice. The caller
loops over batch elements. This is correct because:
1. `batch_size` is typically 1–8 in speculative decoding; the loop overhead
   is negligible compared to the V=32k–128k kernel work.
2. A batched C ABI would complicate the kernel and the ctypes binding
   significantly for minimal gain.

## Denom guard alignment

`flashspec/sampling/rejection.py` uses `_MIN_PROB = 1e-9`.
`flashspec-asm-kernel` uses `_DENOM_GUARD = 1e-9` (reference_op.py).
The C kernel uses `1e-9f` (shim.c).
All three match. No change required.

## `pyproject.toml` change

```toml
[project.optional-dependencies]
asm = [
    'flashspec-asm-kernel>=0.1.1,<2.0.0; sys_platform == "linux" and platform_machine == "x86_64"',
]
```

- `>=0.1.1`: 0.1.0 on PyPI is an sdist without the shared library; it
  cannot work. 0.1.1 is the first release with a binary wheel
  (`py3-none-manylinux2014_x86_64`). 0.1.2 adds argument validation (see
  the routing section); any `>=0.1.1` resolves to it once published.
- Environment marker: the only wheel is Linux x86-64. Elsewhere pip would
  fall back to the sdist, whose build needs nasm and x86-64 Linux and
  fails loudly by design, so `pip install flashspec[asm]` would fail on
  macOS/ARM. With the marker, the extra is simply a no-op there.
- `<2.0.0` guards against future major ABI bumps. The runtime ABI check
  (`abi_version()`) provides a second safety layer.

## Files changed in the flashspec PR

| File | Change |
|---|---|
| `flashspec/sampling/rejection.py` | Optional import block + routing logic in `_sample_residual` |
| `pyproject.toml` | Add `[project.optional-dependencies] asm = [...]` |

No other files touched. The Triton kernel, engine, bandit, metrics, and utils
modules are entirely unchanged.

## What does NOT change

- Default `pip install flashspec` behaviour
- GPU Triton path (unaffected — kernel is CPU-only)
- The mathematical result (ASM kernel tested at ATOL=1e-5 against the
  same reference as the pure-Python path)
- AGENTS.md §2.1 invariant ("residual distribution is immutable") —
  the computation is the same, only the executor changes

## Testing the integration

The flashspec test suite should gain one integration test:

```python
# tests/test_asm_integration.py  (to be added to flashspec repo)
def test_rejection_sample_asm_matches_python(monkeypatch):
    """ASM and pure-Python paths produce identical token distributions."""
    import flashspec.sampling.rejection as rej
    # Force ASM path
    monkeypatch.setattr(rej, "_ASM_AVAILABLE", True)
    # ... run rejection_sample and compare outputs
```

## Risks

| Risk | Mitigation |
|---|---|
| ABI mismatch between independently released packages | `abi_version()` check + `<2.0.0` upper bound |
| ASM kernel used on GPU tensor (wrong device) | `device.type == "cpu"` guard |
| Non-contiguous tensor from upstream reshape | `is_contiguous()` guard |
| Non-float32 tensor (float64, bf16, fp16) | `dtype == torch.float32` guard on p and q; kernel >= 0.1.2 raises `ValueError` instead of writing to a copy |
| Kernel 0.1.0 installed (no shared library) | `>=0.1.1` pin; `OSError` caught alongside `ImportError` |
| Install on non-x86 / non-Linux platforms | Environment marker on the extra; `ImportError` caught at runtime |
| Numerical divergence from NaN inputs | Documented in ADR 0001; both paths produce defined output |
