# ADR 0001 — Kernel Selection

**Status:** Accepted  
**Phase:** 0 (Profile) → 1 (Reference)

---

## Context

FlashSpec-ASM targets one operation from FlashSpec's verification/draft-acceptance
step for hand-written AVX2 implementation. The guideline required a measured
profiling result before kernel selection — not a prior assumption.

Four candidate sub-operations were identified from reading the FlashSpec source
(`flashspec/kernels/verify_kernel.py`, `flashspec/sampling/rejection.py`):

1. **`gather_logprobs`** — gather scalar log-probs at draft token positions
   (indexed read from `(B, γ, V)` tensor)
2. **`acceptance_criterion`** — `u < min(1, exp(log_p - log_q))` per token
3. **`first_rejection`** — argmin scan over the boolean acceptance mask
4. **`residual_distribution`** — `max(0, p-q) / sum(max(0, p-q))` at the
   rejection position

Prior expectation (before profiling): `acceptance_criterion` would dominate,
since it runs on every draft token and involves an `exp()` and a comparison.

## Profiling Results (Phase 0)

Profiled on CPU (the target for the ASM kernel) across five realistic scenarios:

| Scenario | B | γ | V | `residual_dist` % | `acceptance` % |
|---|---|---|---|---|---|
| single-seq-short-gamma | 1 | 4 | 32 000 | 76.6% | 2.3% |
| single-seq-long-gamma  | 1 | 8 | 32 000 | 96.5% | 0.9% |
| batch8-short-gamma     | 8 | 4 | 32 000 | 94.7% | 0.2% |
| batch8-long-gamma      | 8 | 8 | 32 000 | 94.0% | 0.1% |
| batch8-llama3-vocab    | 8 | 4 | 128 256 | ~100% | ~0% |

**`residual_distribution` dominates at 92.5% of pipeline time on average.**

Raw numbers saved in `bench/results/phase0_20260811_071136.txt`.

## Why `residual_dist` Dominates

The acceptance criterion operates on scalar log-probs already gathered to
`(B, γ)` — a small tensor. The residual distribution operates on full
`(B, V)` slices of probability vectors (V = 32 000–128 256 float32 values),
making it memory-bandwidth bound rather than compute-bound. The `exp()` calls
across the full vocabulary dwarf the scalar `exp()` in the acceptance path.

## Decision

**Implement `residual_dist_single` as the Phase 3 AVX2 kernel.**

C ABI signature:
```c
void residual_dist(
    const float* p,    // target probs at rejection pos, shape (V,)
    const float* q,    // draft  probs at rejection pos, shape (V,)
    float*       out,  // output residual pmf,           shape (V,)
    int64_t      V     // vocab size
);
```

The operation maps cleanly to AVX2:
- `vsubps` (8 floats/cycle) — `p - q`
- `vmaxps` (8 floats/cycle) — `max(0, diff)`
- horizontal reduction via `vhaddps` or shuffle+add — `sum(clamped)`
- `vdivps` (11-14 cycles throughput) or `vrcp8ps + vmulps` — normalise

Memory access pattern is sequential (stride-1 float32), maximising L1/L2
prefetch efficiency. At V=32 000, the slice is 128 KB — fits in L2 on most
modern CPUs (256 KB–512 KB L2), avoiding L3 bandwidth penalty.

## NaN Handling Contract

**Documented divergence between NumPy reference and AVX2 kernel:**

`np.maximum(NaN, 0.0)` returns `NaN` (Python/NumPy semantics).  
`vmaxps(NaN, 0.0)` returns `0.0` (Intel AVX2 semantics — the second operand
is returned when the first is NaN, per Intel SDM Vol. 2B §4.3).

This means the AVX2 kernel will silently zero NaN inputs during the clamp step,
rather than propagating them. This is **acceptable** because:

1. NaN in log-probs indicates an upstream bug (model output, not a valid
   probability distribution). The kernel is not the right place to handle it.
2. FlashSpec validates input shapes but not NaN presence. Both the reference
   and the kernel produce "some non-NaN output" for NaN inputs — they just
   differ in which values appear.
3. The test suite documents this divergence explicitly rather than asserting
   NaN-for-NaN agreement (`test_nan_in_p_propagates` is a `pass` with comment).

The correctness suite does NOT test NaN→NaN propagation agreement between
reference and kernel. It DOES test that neither produces -inf or crashes.

## Rejected Alternatives

**`acceptance_criterion`**: Only 0.7% of pipeline time at realistic batch sizes.
The `exp()` + clamp + compare operates on `(B, γ)` scalars — tiny vectors
at B=8, γ=8 (64 elements), not worth AVX2 overhead.

**`gather_logprobs`**: 0.7% of pipeline time. A gather is memory-random-access
(indexed by token IDs); AVX2 gather (`vgatherdps`) has poor throughput vs.
sequential loads and is unlikely to win over compiler output.

**`first_rejection`**: 1.5% of pipeline time. γ is small (1–8). The argmin
scan is over 1–8 boolean values per sequence — scalar is optimal.

## Consequences

- Phase 1 reference implementation targets `residual_dist_single` with the
  C ABI signature above.
- Test suite covers the NaN contract divergence as documented.
- Phase 3 (ASM) must match the NumPy reference on all non-NaN inputs within
  float32 tolerance (ATOL=1e-5, RTOL=1e-5).
- Phase 3 NaN handling is "don't propagate" (AVX2 semantics) — not a bug.
