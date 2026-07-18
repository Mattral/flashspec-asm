# ADR 0003 — AVX2 Baseline vs. AVX-512

**Status:** Accepted  
**Phase:** Project setup

---

## Context

The kernel operates on float32 vectors of length V = 32 000–128 256. Wider
SIMD registers reduce loop iterations and may reduce latency. AVX2 provides
256-bit registers (8 × float32); AVX-512 provides 512-bit registers (16 × float32).

## Decision

**v1 targets AVX2 only.** AVX-512 is a stretch goal gated behind full v1 release.

## Rationale

**Correctness-first:** Wider registers mean a more complex loop epilogue (tail
handling for V % 16 != 0 vs. V % 8 != 0) and additional mask register usage.
Getting the AVX2 path correct and thoroughly tested is the priority. Adding
AVX-512 before the AVX2 path has shipped verified benchmark numbers would
fragment the testing matrix before there's a proven baseline.

**Hardware availability:** AVX2 is available on virtually all x86-64 server
CPUs from the last decade (Intel Haswell+, AMD Zen+). AVX-512 availability
is more fragmented: absent on older Intel server CPUs, absent on some AMD
processors, and subject to frequency throttling on Intel CPUs under sustained
AVX-512 load (thermal downclocking that can erase the throughput gain). A
correct, honest benchmark of AVX-512 vs. AVX2 requires access to specific
hardware; v1 doesn't have that constraint.

**CPUID fallback:** A scalar fallback path (no SIMD) will be compiled in for
safety — selected at runtime if the CPU doesn't report AVX2 support. This is
the conservative choice; in practice every target deployment environment will
have AVX2, but the fallback prevents silent wrong results on unexpected hardware.

## AVX2 Loop Structure for `residual_dist`

```nasm
; rdi = p (float*), rsi = q (float*), rdx = out (float*), rcx = V (int64)
; Strategy: process 8 floats/iteration (one ymm), tail loop for remainder

    vzeroall                    ; clear ymm accumulators
    vxorps  ymm_sum, ymm_sum, ymm_sum  ; accumulate sum for normaliser

.loop:
    vmovups ymm_p,   [rdi + rax*4]
    vmovups ymm_q,   [rsi + rax*4]
    vsubps  ymm_diff, ymm_p, ymm_q     ; diff = p - q
    vmaxps  ymm_cl,  ymm_diff, ymm_zero ; clamp = max(0, diff)
    vaddps  ymm_sum, ymm_sum,  ymm_cl  ; accumulate
    vmovups [rdx + rax*4], ymm_cl      ; store clamped (un-normalised)
    add     rax, 8
    cmp     rax, rcx
    jl      .loop

; Horizontal sum of ymm_sum → scalar denom
; Normalisation pass: divide stored values by denom
; (Two-pass: store clamped first, normalise in second pass)
```

Two-pass approach chosen over fused pass because:
1. The denominator must be complete before any division can occur.
2. The normalisation pass is a simple load+divide+store loop — the bottleneck
   is the first pass (compute-heavy: vsubps + vmaxps + vaddps).
3. The clamped values fit in L2 cache (128 KB for V=32 000), so the second
   pass is not bandwidth-limited in practice.

An alternative single-pass approach using an in-register accumulator and a
second sweep is equivalent but more complex to implement and reason about
for a correctness-first phase.

## Rejected Alternatives

**AVX-512 (zmm registers, 16 × float32):** Would halve the loop iterations
but adds mask-register complexity and potential frequency throttling. Post-v1
stretch goal. A CPUID dispatch function (`cpuid.c`) will be included in the
repo structure from the start so this path can be added without restructuring.

**SSE2/SSE4 (128-bit, 4 × float32):** No reason to target an older baseline
when AVX2 is universally available on the target deployment platforms. The
shim's `cpu_supports_avx2()` check handles the (essentially theoretical)
fallback case.

## CPUID Dispatch Plan (Phase 7)

```c
// cpuid.c — included in v1 shim but AVX-512 path not populated until stretch
int cpu_supports_avx2(void) {
    // Check CPUID.7.0:EBX bit 5 (AVX2)
}
int cpu_supports_avx512f(void) {
    // Check CPUID.7.0:EBX bit 16 (AVX-512F)
}
```

The Python binding (`flashspec_asm.py`) calls `cpu_supports_avx2()` at import
time and falls back to the NumPy/PyTorch path if it returns 0.
