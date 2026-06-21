/*
 * FlashSpec-ASM — C shim (Phase 2 naive baseline + Phase 4 ABI boundary)
 *
 * Two roles:
 *   1. Phase 2: naive C implementation of residual_dist compiled -O3
 *      -march=native. This is the baseline the ASM kernel must beat.
 *   2. Phase 4: thin wrapper around the ASM kernel that handles stack
 *      alignment, CPUID dispatch, and the ABI versioning check.
 *
 * Build:
 *   cc -O3 -march=native -fPIC -c src/shim.c -I include/ -o build/shim.o
 *
 * ABI: System V AMD64. This file is compiled by cc, not assembled by NASM,
 * so calling conventions are handled by the compiler.
 */

#include "flashspec_asm.h"
#include <stdint.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * ABI version — bump when the C signature changes
 * flashspec checks this at import time and falls back on mismatch.
 * ------------------------------------------------------------------------- */
int abi_version(void) {
    return FLASHSPEC_ASM_ABI_VERSION;
}

/* -------------------------------------------------------------------------
 * Phase 2: naive C baseline
 *
 * Intentionally simple: let -O3 -march=native auto-vectorise.
 * This is the honest compiler-optimised baseline for the benchmark.
 * Do NOT hand-optimise this function — it defeats the comparison.
 * ------------------------------------------------------------------------- */
void residual_dist_c(
    const float* restrict p,
    const float* restrict q,
    float*       restrict out,
    int64_t      V
) {
    /* Pass 1: diff = max(0, p - q); accumulate sum */
    float sum = 0.0f;
    for (int64_t v = 0; v < V; ++v) {
        float diff = p[v] - q[v];
        float clamped = diff > 0.0f ? diff : 0.0f;
        out[v] = clamped;
        sum   += clamped;
    }

    /* Denom guard — matches reference_op.py _DENOM_GUARD = 1e-9 */
    if (sum < 1e-9f) sum = 1e-9f;

    /* Pass 2: normalise */
    float inv_sum = 1.0f / sum;
    for (int64_t v = 0; v < V; ++v) {
        out[v] *= inv_sum;
    }
}

/* -------------------------------------------------------------------------
 * Phase 4: dispatch wrapper (populated after Phase 3 ships)
 *
 * Calls the AVX2 kernel if supported, falls back to residual_dist_c.
 * Stack is guaranteed 16-byte aligned by the SysV ABI at the call site;
 * the shim does not need to realign before calling the ASM kernel.
 * ------------------------------------------------------------------------- */
void residual_dist(
    const float* restrict p,
    const float* restrict q,
    float*       restrict out,
    int64_t      V
) {
    if (cpu_supports_avx2()) {
        residual_dist_avx2(p, q, out, V);
    } else {
        residual_dist_c(p, q, out, V);
    }
}
