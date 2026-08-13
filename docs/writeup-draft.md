# FlashSpec-ASM: Hand-writing an AVX2 Kernel for Speculative Decoding

*Status: complete — all phases shipped. This draft is ready for Medium publication.*

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
hand-written AVX2 kernel, tested with property-based tests before a single benchmark
was run, benchmarked honestly against compiler-optimised C, with every non-obvious
decision written up in an ADR. Not "I beat Triton."

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
AVX2: the 8-wide `vsubps` + `vmaxps` + horizontal-sum + `vmulps` loop reduces the
iteration count by 8× versus scalar, and sequential stride-1 access lets hardware
prefetch fill the pipeline.

---

## Phase 0: Profiling

Profiler: `time.perf_counter_ns` on CPU (torch 2.13.0, numpy 2.4.4),
500 iterations per op, 5 scenarios. Full data: `bench/results/phase0_*.txt`.

My prior assumption going in was that `acceptance_criterion` would dominate — it runs
on every draft token and involves an `exp()` call. It was wrong. The per-scenario
breakdown:

| Scenario | residual_dist % | acceptance % |
|---|---|---|
| B=1, γ=4, V=32k | 76.6% | 2.3% |
| B=1, γ=8, V=32k | 96.5% | 0.9% |
| B=8, γ=4, V=32k | 94.7% | 0.2% |
| B=8, γ=8, V=32k | 94.0% | 0.1% |
| B=8, γ=4, V=128k | ~100% | ~0% |

The reason the acceptance criterion barely registers: it operates on `(B, γ)` scalar
log-probs already gathered — at most 64 floats. The residual operates on `(B, V)`
probability slices — 32k to 128k floats. The vocabulary dimension swamps everything
else.

This is why you measure before you commit to a kernel. The assumption was wrong by
two orders of magnitude.

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
case, V=128256 large vocab, point masses, extreme magnitudes near float32 subnormal,
and non-multiples of 8 (V=7, 9, 15, 16, 17) — the size class that caught the Phase 3
tail bug.

**Pipeline smoke tests:** shape contracts, first_rejection bounds, forced all-accept
(u=0) and all-reject (u=1) paths.

34 tests. All green before Phase 2 was started.

One thing that came out of Phase 1: `vmaxps(NaN, 0)` is a real semantic difference
between NumPy and AVX2. NumPy returns NaN per Python semantics; AVX2 returns 0.0 per
Intel SDM Vol. 2B §4.3. I documented this in ADR 0001 and left a named `pass` test
with a comment rather than papering over the divergence. The kernel's behaviour is
correct for this use case — NaN in a log-prob vector means an upstream fault, not a
valid probability state — but undocumented divergences from the reference are how
production bugs hide.

---

## Phase 2: C baseline

Naive C, compiled `-O3 -march=native -fPIC`. Two-pass: accumulate clamped diffs into
the output buffer, then normalise in a second pass. Deliberately not hand-optimised —
the point is to see what a competent compiler does automatically.

GCC 13.3 auto-vectorised both loops to 32-byte ymm registers (confirmed via
`-fopt-info-vec`). But inspecting the generated assembly (`-S` output) revealed two
specific weaknesses:

**1. Horizontal reduction:** GCC serialises the ymm accumulator via repeated scalar
`vaddss` — 8 sequential additions — rather than a `vhaddps` + `vextractf128` +
`vaddss` tree that collapses 8 partial sums in 3 instructions.

**2. Normalisation pass:** GCC emits scalar `vdivss` (~14 cycles throughput, 1
element) then `vbroadcastss`, rather than `vbroadcastss` of the reciprocal followed
by `vmulps` (~5 cycles throughput, 8 elements).

Baseline numbers:

| V | C mean | C median |
|---|---|---|
| 1,024 | 2,650 ns | 2,366 ns |
| 4,096 | 7,286 ns | 6,489 ns |
| 32,000 | 42,993 ns | 41,334 ns |
| 128,256 | 184,315 ns | 174,487 ns |

These are the numbers the kernel has to beat. They were recorded and checked into
`bench/results/phase2_*.txt` before Phase 3 was started.

---

## Phase 3: The AVX2 kernel

Two-pass, matching the C structure but fixing both identified weaknesses.

**Pass 1:** 8 floats per iteration via ymm registers. The core loop body:

```nasm
vmovups ymm0, [rdi + rax*4]    ; load p[i..i+7]
vmovups ymm1, [rsi + rax*4]    ; load q[i..i+7]
vsubps  ymm2, ymm0, ymm1       ; diff = p - q
vmaxps  ymm3, ymm2, ymm5       ; clamp = max(0, diff)  [ymm5 = 0.0]
vaddps  ymm4, ymm4, ymm3       ; accumulate into ymm sum
vmovups [rdx + rax*4], ymm3    ; store clamped (un-normalised)
add     rax, 8
```

**Horizontal reduction (the targeted fix):** Rather than GCC's 8 scalar `vaddss`,
the kernel uses a 3-instruction tree:

```nasm
vhaddps ymm4, ymm4, ymm4       ; pairwise within each 128-bit lane
vhaddps ymm4, ymm4, ymm4       ; pairs again
vextractf128 xmm0, ymm4, 1     ; upper lane
vaddss  xmm0, xmm0, xmm4      ; final scalar total in xmm0
```

**Normalisation (the second fix):** Reciprocal broadcast + multiply instead of
scalar divide:

```nasm
vdivss  xmm0, [one_f], xmm6    ; 1.0 / sum  (one scalar divide)
vbroadcastss ymm7, xmm0        ; broadcast to all 8 lanes
; ... pass 2 loop:
vmulps  ymm0, ymm0, ymm7       ; 8-wide multiply
```

**Tail handling:** Scalar loop for `V % 8` remainder elements, using a
*separate* `xmm6` accumulator — not `xmm4`/`ymm4`. This is where the bug was.

**`vzeroupper` on exit** to prevent the AVX→SSE transition penalty on Intel
microarchitectures (Haswell/Broadwell particularly).

### The bug I found

**v1** accumulated tail elements via `vaddss xmm4, xmm4, xmm3`. This writes to
`xmm4`, which is the low 128-bit lane of `ymm4` — the vectorised accumulator. The
horizontal reduction then summed all 8 lanes of `ymm4`, double-counting the
vectorised partial sums already stored in the upper 7 lanes.

Symptom: outputs summed to > 1.0 at V=9, V=15, V=17. Caught immediately by the
correctness suite. The fix (v2): use `xmm6` as a completely separate scalar
accumulator; combine with the hreduce result via `vaddss xmm6, xmm6, xmm0` after
the reduction.

This is the class of bug that correctness-first discipline exists to catch. A fast
wrong kernel is strictly worse than no kernel.

---

## Benchmark results

Three-way: NumPy reference, GCC 13.3 `-O3 -march=native`, hand-written AVX2 NASM.
Platform: Linux x86-64, AVX2 confirmed. `perf_counter_ns`, 50 warmup + 2000 iterations.

| V | NumPy | C -O3 | AVX2 | vs NumPy | vs C |
|---|---|---|---|---|---|
| 1,024 | 7,738 ns | 2,650 ns | 1,570 ns | 4.93× | **1.69×** |
| 4,096 | 11,959 ns | 7,250 ns | 2,233 ns | 5.36× | **3.25×** |
| 32,000 | 59,906 ns | 42,993 ns | 8,865 ns | 6.76× | **4.85×** |
| 128,256 | 248,894 ns | 184,315 ns | 42,464 ns | 5.86× | **4.34×** |

**Why the gains scale with V:** At V=1024, loop setup and the tail handler are a
non-trivial fraction of total time. At V=32000–128256, the vectorised loop dominates
and the two targeted GCC weaknesses account for a larger share of the cycle budget.
The V=128256 case is slightly less efficient than V=32000 because the 500 KB output
buffer exceeds L2 on most CPUs (~256–512 KB), adding L3 latency on the pass 2 sweep.

**What was forecast vs. what happened:** ADR 0004 conservatively predicted V=1024
might not beat C due to loop overhead. It beat C by 1.69×. The forecast was wrong in
the optimistic direction — worth noting because honest forecasting is part of the
methodology.

**For production measurement:** re-run with `taskset -c 0 perf stat -e cycles,
instructions,cache-misses` to pin the process and count hardware events. The
`perf_counter_ns` numbers include scheduler jitter and are adequate for relative
comparison but not absolute cycle accounting.

---

## What I'd do differently

**Measure the horizontal reduction in isolation.** The `vhaddps` tree operates within
128-bit lanes and requires an `vextractf128` step — a minor but real latency hit on
some microarchitectures. An alternative using `vpermf32` + `vaddps` shuffles might
be faster on specific CPUs. I didn't microbenchmark the reduction separately; I
measured the full kernel. For a production kernel that needs to squeeze the last few
percent, the reduction deserves its own `rdtsc` harness.

**Consider alignment.** The kernel uses `vmovups` (unaligned loads) throughout. On
modern CPUs with cache-line-aligned inputs, `vmovups` and `vmovaps` have identical
throughput. But callers passing non-aligned buffers pay a penalty. A 32-byte alignment
requirement on `p`, `q`, and `out` — enforced in the Python binding with
`np.ascontiguousarray` + a posix_memalign path — would let the kernel use `vmovaps`
safely and document the contract explicitly.

**Ship the `perf stat` run as part of CI.** The `perf_counter_ns` methodology is
reproducible but noisy. A CI job that runs `perf stat` with process pinning would
give cycle-accurate numbers across builds, making performance regressions visible
before they reach main. This requires a non-containerised runner with `perf` access
— not free, but worth it for a kernel repo.

**Write the fuzz target.** The `tests/fuzz/` directory exists but is empty.
A `libFuzzer` harness comparing the C baseline and ASM kernel byte-by-byte on random
inputs would catch any remaining edge-case divergence that Hypothesis didn't surface.
It's a 50-line C file; there's no good reason it isn't there.

---

## Related

- [FlashSpec on PyPI](https://pypi.org/project/flashspec/)
- [KANX](https://github.com/Mattral/KANX) — same writeup format
- ADR 0001–0004 in `docs/adr/`
