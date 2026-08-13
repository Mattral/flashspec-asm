# ADR 0004 — C Baseline Analysis and Phase 3 Target

**Status:** Accepted  
**Phase:** 2 (C Baseline) → 3 (ASM Kernel)

---

## Context

Phase 2 exit criterion: compile `residual_dist_c` with `-O3 -march=native`,
benchmark it against the NumPy reference, and record the numbers before
writing any assembly. This ADR documents what the compiler actually produced
and where the hand-written kernel has room to improve.

## What GCC 13.3 auto-vectorised

Both loops in `residual_dist_c` were auto-vectorised to 32-byte (AVX2 ymm)
vectors per GCC's `-fopt-info-vec` output:

```
shim.c:44 (pass 1 — clamp+accumulate): vectorized using 32-byte vectors
shim.c:56 (pass 2 — normalise):        vectorized using 32-byte vectors
```

However, inspection of the generated assembly (`-S` output) reveals two
specific weaknesses:

**1. Horizontal reduction (pass 1 tail):**  
GCC reduces the ymm accumulator to a scalar sum via repeated scalar `vaddss`
instructions rather than a `vhaddps` + shuffle sequence. This serialises 8
additions that could be done in 3 instructions (log₂8 tree).

**2. Normalisation scalar (pass 2):**  
GCC broadcasts the scalar divisor with `vbroadcastss` then uses scalar
`vdivss` rather than `vdivps` on the full ymm register. Division is the
most expensive operation (~11–14 cycle throughput); vectorising it 8-wide
gives a direct throughput gain.

## Measured baseline numbers

| Scenario | C -O3 mean | C -O3 median | vs NumPy |
|---|---|---|---|
| V=1024 | 2589 ns | 2366 ns | 3.03× faster |
| V=4096 | 7286 ns | 6489 ns | 1.80× faster |
| V=32000 | 43295 ns | 41334 ns | 1.27× faster |
| V=128256 | 181431 ns | 174487 ns | 1.37× faster |

Raw results: `bench/results/phase2_c_baseline.txt`

**The C baseline is already fast.** At large V (32k–128k), both NumPy and C
are memory-bandwidth bound — the gap shrinks because the bottleneck shifts
from compute to DRAM throughput. The hand-written kernel will help most if it
keeps the working set in L2 (128 KB for V=32k at float32) and eliminates the
scalar reduction and normalise instructions.

## Phase 3 targets

**Primary target: V=32000 and V=128256** — the realistic deployment sizes.
The hand-written kernel aims to:
1. Replace the scalar horizontal reduction with a 3-instruction ymm tree
   (`vhaddps` × 2 + `vaddps` on extracted lanes).
2. Replace scalar `vdivss` with `vdivps` (8-wide) in the normalise pass
   via `vbroadcastss` of `1/denom` + `vmulps`.

**Small V (1024):** The ASM kernel may not beat C here. Loop setup and
tail handling overhead are non-trivial relative to the work done at 1024
elements. If it loses, document it plainly.

## Honesty clause

If the hand-written kernel does not beat the C baseline on any scenario,
the Phase 5 writeup will say so, with the actual numbers and an explanation.
Per the project guideline: "a marginal or negative speedup is published
honestly — the correctness/methodology story is the real deliverable."

## Decision

**Proceed to Phase 3.** The two identified weaknesses (scalar hreduce, scalar
vdivss) give the hand-written kernel concrete, measurable targets. The baseline
numbers are recorded and checked in. The correctness suite is green.
