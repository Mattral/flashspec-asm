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
except ImportError:
    pass
```

**Runtime routing conditions** — ASM path used only when ALL of:
- `_ASM_AVAILABLE` is True (kernel installed, ABI matches, AVX2 present)
- `p_at_rej.device.type == "cpu"` (never intercept GPU tensors)
- `p_at_rej.is_contiguous()` (kernel requires contiguous float32 memory)

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
asm = ["flashspec-asm-kernel>=0.1.0,<2.0.0"]
```

The `<2.0.0` upper bound guards against future major ABI bumps. The runtime
ABI check (`abi_version()`) provides a second safety layer.

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
| Import error on non-x86 platforms (ARM, etc.) | `ImportError` silently caught |
| Numerical divergence from NaN inputs | Documented in ADR 0001; both paths produce defined output |
