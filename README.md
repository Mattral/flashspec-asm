# flashspec-asm

Hand-written x86-64 AVX2 assembly kernel for the residual distribution
computation in speculative decoding — shipped as an optional accelerated
backend for [FlashSpec](https://pypi.org/project/flashspec/).

**This is a systems-programming portfolio project first, a performance project
second.** The honest pitch: hand-written production assembly, tested and
benchmarked against a real compiler-optimised C baseline. It is not "N× faster
than Triton" — Triton runs on GPU; this targets the CPU fallback path only.

---

## What this is

Speculative decoding ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192))
accelerates autoregressive inference by letting a small draft model propose
γ tokens which a target model then accepts or rejects in one forward pass.
At the first rejection position the target distribution must be repaired to
a valid residual probability mass function before the next token is sampled:

```
residual[v] = max(0, p[v] − q[v])          # relu of per-token difference
residual    = residual / Σ residual          # renormalise to a valid pmf
```

FlashSpec uses a Triton kernel for this on GPU. On the CPU fallback path,
Phase 0 profiling revealed that `residual_distribution` accounts for **92.5%
of verification pipeline time on average** — ranging from 76.6% at
`(B=1, γ=4, V=32000)` to effectively 100% at `(B=8, γ=4, V=128256)`.
Full profiling data: [`bench/results/phase0_20260811_071136.txt`](bench/results/phase0_20260811_071136.txt).

This repo hand-implements that single operation in AVX2 NASM assembly, exposes
it through a C ABI shim, and wires it into FlashSpec as an opt-in backend.

## What this is NOT

- A port of the FlashSpec engine — bandit logic, orchestration, Triton kernels
  and draft selection all remain untouched in the main package
- A GPU accelerator — this targets CPU only; the Triton path is faster on GPU
  and is not replaced
- Guaranteed to beat compiler-optimised C — the benchmark section is honest
  about where and why it wins, and where it does not

---

## Why not just the compiler?

GCC 13.3 with `-O3 -march=native` auto-vectorises both loops in `residual_dist`
to 32-byte ymm registers (confirmed via `-fopt-info-vec`). But inspecting the
generated assembly (`-S` output) reveals two specific weaknesses:

**1. Horizontal reduction (pass 1):** GCC serialises the ymm accumulator via
repeated scalar `vaddss` instructions — 8 sequential additions — rather than a
`vhaddps` + `vextractf128` + `vaddss` tree that collapses 8 partial sums in
3 instructions.

**2. Normalisation pass (pass 2):** GCC emits a scalar `vdivss` (1 element
per ~14 cycles) followed by a `vbroadcastss`, rather than `vbroadcastss` of
the reciprocal followed by `vmulps` (8 elements in ~5 cycles for the
multiply).

The hand-written kernel targets both weaknesses explicitly. ADR 0004 documents
the `-S` analysis and the measured baseline before any assembly was written.

---

## Architecture

### Algorithm: two-pass, not fused

A fused pass that computes the normalised output in one sweep is tempting but
impossible: the denominator must be complete before any division can occur.
The two-pass design stores the un-normalised clamped values in the output
buffer during pass 1, computes the denominator, then multiplies in pass 2.

At `V=32000`, the output buffer (128 KB of float32) fits in L2 cache on most
modern CPUs (256–512 KB L2), so pass 2 is not DRAM-bandwidth bound. At
`V=128256` (LLaMA-3 vocabulary) the buffer is 500 KB and will incur L3
latency on the second pass — this is visible in the benchmark numbers.

### Register allocation

```
rdi, rsi, rdx, rcx   arguments: p*, q*, out*, V  (SysV AMD64)
r8                   main loop bound: V & ~7
rax                  loop index
ymm0, ymm1           p and q chunks (8 × float32 each)
ymm2                 diff = vsubps(p, q)
ymm3                 clamped = vmaxps(diff, 0)
ymm4                 ymm accumulator — vectorised partial sums (8 lanes)
ymm5                 zero constant for vmaxps clamp
xmm6                 scalar tail accumulator + final collapsed sum
ymm7                 broadcast 1/denom for pass 2 normalise
```

ymm registers 0–7 are caller-saved under SysV AMD64; the kernel uses only
these, avoiding callee-save overhead. `vzeroupper` is emitted before `ret`
to prevent the AVX→SSE transition penalty on Intel microarchitectures.

### NaN handling contract (documented divergence)

`np.maximum(NaN, 0.0)` returns `NaN` (Python semantics).
`vmaxps(NaN, 0.0)` returns `0.0` (Intel SDM Vol. 2B §4.3 — the second operand
is returned when the first is a quiet NaN).

The AVX2 kernel silently zeroes NaN inputs during the clamp step rather than
propagating them. This is **intentional and documented**, not a bug:

1. NaN in a probability vector indicates an upstream fault (invalid model
   output); the rejection-sampling kernel is not the right recovery point.
2. FlashSpec validates input shapes but not NaN presence; the kernel's
   behaviour matches production use where inputs are well-formed.
3. The correctness suite documents this divergence explicitly —
   `test_nan_in_p_propagates` is a `pass` with a comment, not an assertion.

The suite *does* assert that neither the reference nor the kernel produces
`-inf` or a runtime fault on NaN input.

---

## Project status

| Phase | Description | Status |
|---|---|---|
| 0 — Profile | Identify hot-path op with measured cycle share | ✅ Done |
| 1 — Reference | NumPy/PyTorch reference + property-based tests | ✅ Done |
| 2 — C baseline | Naive C, `-O3 -march=native`, benchmarked | ✅ Done |
| 3 — ASM kernel | Hand-written AVX2 NASM | ✅ Done |
| 4 — FFI layer | C shim + Python `ctypes` bindings | ✅ Done |
| 5 — Benchmark | Cycle-level C vs ASM comparison | ✅ Done |
| 6 — Release | ADRs, README, GitHub release | ✅ Done |

See [`docs/roadmap.md`](docs/roadmap.md) for phase exit criteria and the
FlashSpec integration architecture (optional `pip install flashspec[asm]`).

---

## Build

```bash
# Requirements: nasm ≥ 2.14, cc (gcc or clang), python3
make          # → libflashspec_asm.so
make test     # correctness suite (must be green before benchmarking)
make bench    # three-way benchmark: NumPy vs C vs ASM
make profile  # re-run Phase 0 profiler
```

CI (GitHub Actions) assembles, runs the correctness suite on every push and
pull request, and fails on any correctness regression. Benchmark regressions
are reported but do not fail CI — numbers vary across runner hardware.

### Python

```bash
pip install hypothesis pytest numpy torch
pytest tests/ -v
```

The test suite covers: valid-pmf output contract, zero-where-q-dominates,
batched/single-slice agreement, NaN/inf propagation (documented divergence),
denom-guard (all-zero residual path), V=1 through V=128256, point masses,
all-accepted and all-rejected forcing, and non-multiples of 8 (V=7, 9, 15,
16, 17) — the class of size that exposed the tail-accumulator bug below.

---

## Benchmark results

Three-way comparison: NumPy reference, naive C (`GCC 13.3 -O3 -march=native
-fPIC`), and hand-written AVX2 NASM. Platform: Linux x86-64, AVX2 confirmed.
Measurement: `perf_counter_ns`, 50 warmup iterations + 2000 timed iterations.
Input: single `(V,)` float32 slice, Dirichlet-sampled, seed 0.

| Vocab size | NumPy (mean) | C −O3 (mean) | ASM (mean) | vs NumPy | vs C |
|---|---|---|---|---|---|
| V=1,024 | 7,738 ns | 2,650 ns | 1,570 ns | 4.93× | 1.69× |
| V=4,096 | 11,959 ns | 7,250 ns | 2,233 ns | 5.36× | 3.25× |
| V=32,000 *(LLaMA-2)* | 59,906 ns | 42,993 ns | 8,865 ns | 6.76× | 4.85× |
| V=128,256 *(LLaMA-3)* | 248,894 ns | 184,315 ns | 42,464 ns | 5.86× | 4.34× |

Dated raw results (mean, median, per-size): [`bench/results/phase5_20260812.txt`](bench/results/phase5_20260812.txt).

**Why the gains scale with V:** At small V (1,024), loop setup and tail
handling are a non-trivial fraction of total time, compressing the headroom.
At large V (32,000–128,256), the vectorised loop dominates and the targeted
GCC weaknesses — scalar horizontal reduction and scalar `vdivss` — represent
a larger share of the total cycle budget. The V=128,256 case shows a slight
efficiency drop relative to V=32,000 because the 500 KB output buffer exceeds
L2 on most CPUs, adding L3 latency on the pass 2 normalise sweep.

**What was forecast vs. what happened:** ADR 0004 conservatively predicted
that V=1,024 might not beat the C baseline due to loop overhead. It did
(1.69×). Honesty requires noting this too — the forecast was wrong in the
optimistic direction.

**For a production cycle-accurate measurement:** re-run with
`taskset -c 0 perf stat -e cycles,instructions,cache-misses ./bench/bench_harness`
to pin the process and count hardware events. The `perf_counter_ns` numbers
above include scheduler jitter.

---

## Bugs found

### Phase 3, v1 — tail accumulator corruption (caught by correctness suite)

**Symptom:** Correctness suite failed at V=9, V=15, V=17 (any V where
V % 8 ≠ 0). Output summed to > 1.0.

**Root cause:** The pass 1 tail loop accumulated scalar remainder elements
using `vaddss xmm4, xmm4, xmm3` — writing to `xmm4`, which is the low
128-bit lane of `ymm4`, the vectorised accumulator. The horizontal reduction
then summed all 8 lanes of `ymm4`, which now contained 7 lanes of legitimate
vectorised partial sums *plus* the tail sum in lane 0, causing the tail
elements to be counted twice.

**Fix (v2):** Tail elements are accumulated in `xmm6`, a register entirely
separate from `ymm4`. The horizontal reduction collapses `ymm4` to a scalar
in `xmm0`, then `vaddss xmm6, xmm6, xmm0` combines the two accumulators.
All 13 cross-size correctness cases pass at ATOL=1e-5.

This is the class of bug that correctness-first discipline exists to catch.
A fast wrong kernel is strictly worse than no kernel.

---

## Architecture decisions

- [ADR 0001 — Kernel selection](docs/adr/0001-kernel-selection.md) — profiling
  methodology, why `residual_dist` was chosen over `acceptance_criterion`
  (prior hypothesis), and the NaN handling contract
- [ADR 0002 — Assembler and ABI](docs/adr/0002-assembler-and-abi-choice.md) —
  NASM / Intel syntax / SysV AMD64 rationale; `vzeroupper` requirement
- [ADR 0003 — AVX2 vs. AVX-512](docs/adr/0003-avx2-baseline-vs-avx512.md) —
  correctness-first argument, AVX-512 frequency throttling on Intel, loop
  epilogue complexity, CPUID dispatch structure for future extension
- [ADR 0004 — C baseline analysis](docs/adr/0004-c-baseline-analysis.md) —
  GCC `-S` output analysis, two identified weaknesses, Phase 3 targets,
  explicit honesty clause

---

## Known limitations

- **Linux x86-64 only** (ELF, SysV AMD64 ABI) — no Windows/MSVC support
  in v1; Microsoft x64 uses a different register argument layout
- **AVX2 required** at runtime; `cpu_supports_avx2()` (CPUID leaf 7, EBX
  bit 5) selects the scalar C fallback automatically if absent
- **CPU path only** — the Triton GPU kernel is unchanged and faster for GPU
  inference; this accelerates the CPU fallback path only
- **Single-slice only** — the kernel processes one `(V,)` slice; batching
  is handled by the caller (loop over batch elements)
- **AVX-512 and ARM NEON** are post-v1 stretch goals; the CPUID dispatch
  infrastructure is already in place for an AVX-512 path
- **`perf_counter_ns` methodology** is adequate for relative comparison but
  not for absolute cycle counts; see the production measurement note above

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

The explicit patent grant in Apache 2.0 matters more for low-level performance
code that may be adopted into production stacks than it does for typical Python
packages. MIT is common in systems-programming repos but Apache 2.0's patent
protection is the right default for infrastructure-adjacent work.

---

## Related

- [FlashSpec](https://pypi.org/project/flashspec/) — adaptive speculative
  decoding engine with bandit-based draft selection; the main package
- [KANX](https://github.com/Mattral/KANX) — production-grade
  Kolmogorov-Arnold Networks library (same author)
- [guardrail-rs](https://github.com/Mattral/guardrail-rs) — production Rust
  LLM security reverse proxy (same author)
