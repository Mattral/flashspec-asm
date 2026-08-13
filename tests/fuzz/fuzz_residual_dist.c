/*
 * FlashSpec-ASM — libFuzzer harness for residual_dist
 *
 * Compares residual_dist_c (C baseline) against residual_dist_avx2 (ASM kernel)
 * on arbitrary byte inputs from the fuzzer, reinterpreted as float32 probability
 * vectors. Any output disagreement above ATOL=1e-4 is reported as a crash.
 *
 * Build and run:
 *   clang -O1 -fsanitize=fuzzer,address -I ../../include \
 *         fuzz_residual_dist.c ../../build/shim.o ../../build/cpuid.o \
 *         ../../build/kernel_avx2.o \
 *         -o fuzz_residual_dist
 *   ./fuzz_residual_dist -max_len=4096 corpus/
 *
 * The harness normalises the input bytes to valid float32 pmfs before calling
 * either function, so NaN/inf inputs from the fuzzer hit the normalisation path
 * rather than exercising the NaN contract directly. That divergence is already
 * documented in ADR 0001 and covered by test_nan_in_p_numpy_contract.
 *
 * Status: stub — compile-tested, not yet run in CI.
 * See writeup §"What I'd do differently" for rationale.
 */

#include "flashspec_asm.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ATOL 1e-4f
#define MIN_V 1
#define MAX_V 512   /* keep corpus small; V=32000 is too slow for fuzzing */

/* Normalise a float32 buffer to a valid pmf in-place.
 * Replaces NaN/inf with 0, abs()s negatives, then divides by sum.
 * If sum == 0 (all-zero input), sets index 0 = 1.0 (degenerate pmf). */
static void make_pmf(float *p, int64_t V) {
    float sum = 0.0f;
    for (int64_t i = 0; i < V; i++) {
        if (!isfinite(p[i]) || p[i] < 0.0f) p[i] = 0.0f;
        sum += p[i];
    }
    if (sum < 1e-9f) { p[0] = 1.0f; sum = 1.0f; }
    for (int64_t i = 0; i < V; i++) p[i] /= sum;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    /* Need at least 2 floats for p and q, plus 1 byte for V selector */
    if (size < sizeof(float) * 2 + 1) return 0;

    /* Derive V from first byte: map [0,255] → [MIN_V, MAX_V] */
    int64_t V = MIN_V + (data[0] % (MAX_V - MIN_V + 1));
    size_t needed = 1 + sizeof(float) * (size_t)(2 * V);
    if (size < needed) return 0;

    float *p   = (float *)malloc(V * sizeof(float));
    float *q   = (float *)malloc(V * sizeof(float));
    float *out_c   = (float *)malloc(V * sizeof(float));
    float *out_asm = (float *)malloc(V * sizeof(float));
    if (!p || !q || !out_c || !out_asm) { free(p); free(q); free(out_c); free(out_asm); return 0; }

    /* Copy fuzzer bytes into p and q, then normalise */
    memcpy(p, data + 1,                  V * sizeof(float));
    memcpy(q, data + 1 + V * sizeof(float), V * sizeof(float));
    make_pmf(p, V);
    make_pmf(q, V);

    /* Run both implementations */
    residual_dist_c(p, q, out_c, V);

    if (cpu_supports_avx2()) {
        residual_dist_avx2(p, q, out_asm, V);

        /* Compare outputs — any disagreement is a bug */
        for (int64_t i = 0; i < V; i++) {
            float diff = out_c[i] - out_asm[i];
            if (diff < 0.0f) diff = -diff;
            if (diff > ATOL) {
                /* libFuzzer treats abort() as a crash and saves the corpus */
                __builtin_trap();
            }
        }
    }

    free(p); free(q); free(out_c); free(out_asm);
    return 0;
}
