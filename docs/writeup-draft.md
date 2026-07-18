# FlashSpec-ASM: Hand-writing an AVX2 Kernel for Speculative Decoding

*Status: draft — sections marked [TBD] are placeholders to fill after each phase completes.*

---

## Why I built this

[FlashSpec](https://pypi.org/project/flashspec/) is an adaptive speculative decoding
engine I shipped earlier this year. It uses a bandit-based draft selector and a Triton
kernel for GPU-side token verification. The GPU path is fast. The CPU fallback path
is not something I'd thought hard about.

This project is the CPU fallback path, taken seriously. Not because I needed to — the
GPU path handles the real workload — but because I wanted to hand-write production
assembly against a real benchmark target, document every decision I made, and publish
the actual numbers, including the ones where the compiler wins.

This is a systems-programming portfolio piece first. The pitch is: here is a
hand-written AVX2 kernel, tested with property-based tests and fuzz before a single
benchmark was run, benchmarked honestly against compiler-optimised C and a real Triton
kernel, with every non-obvious decision written up in an ADR. Not "I beat Triton."

---

## The target operation

Speculative decoding works by generating γ draft tokens cheaply, then verifying them
against the target model in parallel. The verification step uses rejection sampling:
for each draft token, accept it with probability `min(1, p_target / p_draft)`. On
rejection, the algorithm resamples from a *residual distribution*:

```
residual[v] = max(0, p[v] - q[v])
residual    = residual / sum(residual)
```

I profiled all four sub-operations of the verification step across five realistic
parameter scenarios (batch sizes 1–8, speculation depth γ 4–8, vocab sizes 32k–128k).

| Operation | Avg % of pipeline |
|---|---|
| `residual_distribution` | **92.5%** |
| `first_rejection` | 1.5% |
| `acceptance_criterion` | 0.7% |
| `gather_logprobs` | 0.7% |

The residual distribution dominated because it operates on a full `(B, V)` probability
slice — up to 500k float32 values — while the acceptance criterion operates on `(B, γ)`
scalar pairs. At V=128256 (LLaMA-3), the residual step consumed essentially all of the
measured CPU time.

This is memory-bandwidth bound, not compute bound. That profile is exactly right for
AVX2: the 8-wide `vsubps` + `vmaxps` + horizontal-sum + `vdivps` loop reduces the
iteration count by 8× versus scalar, and sequential stride-1 access lets hardware
prefetch fill the pipeline.

---

## Phase 0: Profiling

[TBD — paste real `perf stat` output here once Phase 0 is run on the benchmark machine]

Profiler: `time.perf_counter_ns` on CPU (no GPU), 500 iterations per op, 5 scenarios.
Raw results in `bench/results/phase0_*.txt`.

---

## Phase 1: Reference implementation and test suite

Before writing a single byte of assembly I wrote the ground truth in
`reference/reference_op.py` and a correctness suite in `tests/test_correctness.py`.

The test suite has three layers:

**Property-based (Hypothesis):** 200 examples per property across random float32 inputs.
Checks: output is a valid pmf (non-negative, sums to 1 or 0), zero-where-q-dominates,
scale invariance, batched/single-slice agreement.

**Explicit edge cases:** NaN contract (AVX2 `vmaxps` silently zeros NaN — documented,
not a bug), denom guard (prevents divide-by-zero when all diffs ≤ 0), V=1 degenerate
case, V=128256 large vocab, point masses, extreme magnitudes near float32 subnormal.

**Pipeline smoke tests:** shape contracts, first_rejection bounds, forced all-accept
(u=0) and all-reject (u=1) paths.

34 tests. All green before Phase 2 was started.

One thing I found during Phase 1: the `vmaxps(NaN, 0)` behaviour is a real semantic
difference between the NumPy reference and the AVX2 kernel. NumPy returns NaN;
AVX2 returns 0. I documented this in ADR 0001 and in the test file rather than
papering it over. The kernel's behaviour is correct for this use case (NaN in
log-probs indicates an upstream bug, not a valid probability state), but it's the
kind of thing that bites you if you don't write it down.

---

## Phase 2: C baseline

[TBD — fill in after Phase 2 is complete]

Naive C, compiled `-O3 -march=native`. Two-pass: accumulate clamped diffs into the
output buffer, then normalise in a second pass. Deliberately not hand-optimised —
the point is to see what a competent compiler does automatically.

What the compiler generated: [TBD — paste relevant godbolt / objdump output]

Baseline numbers: [TBD]

---

## Phase 3: The AVX2 kernel

[TBD — fill in after Phase 3 is complete]

Two-pass strategy:

**Pass 1:** Process 8 floats per iteration using ymm registers.
```nasm
vmovups ymm_p,    [rdi + rax*4]
vmovups ymm_q,    [rsi + rax*4]
vsubps  ymm_diff, ymm_p,    ymm_q
vmaxps  ymm_cl,   ymm_diff, ymm_zero
vaddps  ymm_sum,  ymm_sum,  ymm_cl
vmovups [rdx + rax*4], ymm_cl
```

**Horizontal reduction:** [TBD — vhaddps vs. shuffle+add decision]

**Pass 2:** Normalise stored values by the computed denom.

**Tail handling:** Scalar loop for `V % 8` remainder elements.

**`vzeroupper`** on exit to prevent AVX→SSE transition penalty on Intel CPUs.

Bugs found during Phase 3: [TBD — this section will have something in it]

---

## Benchmark results

[TBD — fill in after Phase 5 is complete]

| Scenario | NumPy | C -O3 | AVX2 ASM | vs C |
|---|---|---|---|---|
| V=32k, B=1 | | | | |
| V=32k, B=8 | | | | |
| V=128k, B=8 | | | | |

Hardware: [TBD]
Compiler: [TBD]
Measurement: `perf stat -e cycles,instructions,cache-misses` + `rdtsc`
Methodology: 20 warmup iterations, 1000 measured, taskset -c 0

**Cases where ASM does not win:** [TBD — expected at small V where loop overhead
dominates or the compiler auto-vectorises aggressively]

---

## What I'd do differently

[TBD — written after Phase 6]

---

## Related

- [FlashSpec on PyPI](https://pypi.org/project/flashspec/)
- ADR 0001: Kernel selection
- ADR 0002: Assembler and ABI choice
- ADR 0003: AVX2 baseline vs. AVX-512
- [KANX writeup](https://medium.com/@mattral-lifelong-learning) — same format
