# Fuzz testing — residual_dist

`fuzz_residual_dist.c` is a [libFuzzer](https://llvm.org/docs/LibFuzzer.html)
harness that compares `residual_dist_c` (C baseline) against
`residual_dist_avx2` (ASM kernel) on arbitrary fuzzer-generated inputs.

Any output disagreement above ATOL=1e-4 causes `__builtin_trap()`, which
libFuzzer treats as a crash and saves the triggering input to the corpus.

## Build

```bash
# From repo root — build .so first
make

# Build fuzz target (requires clang with libFuzzer support)
clang -O1 -fsanitize=fuzzer,address \
    -I include/ \
    tests/fuzz/fuzz_residual_dist.c \
    build/shim.o build/cpuid.o build/kernel_avx2.o \
    -lm \
    -o fuzz_residual_dist
```

## Run

```bash
mkdir -p tests/fuzz/corpus
./fuzz_residual_dist -max_len=4096 -max_total_time=300 tests/fuzz/corpus/
```

## Status

Compile-tested. Not yet wired into CI (requires clang + libFuzzer runner).
See `docs/writeup-draft.md §"What I'd do differently"` for rationale on
why this should be added to CI as a future step.

## NaN note

The harness normalises fuzzer bytes to valid pmfs before calling either
function. The NaN handling divergence between NumPy and `vmaxps` is
documented in ADR 0001 and covered by `test_nan_in_p_numpy_contract`
in the main test suite — not exercised here.
