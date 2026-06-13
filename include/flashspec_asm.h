/*
 * flashspec_asm.h — Public C API for flashspec-asm
 *
 * This header defines the stable ABI exposed by libflashspec_asm.so.
 * The Python ctypes bindings (bindings/python/flashspec_asm.py) and the
 * flashspec integration shim both consume this interface.
 *
 * ABI versioning: bump FLASHSPEC_ASM_ABI_VERSION whenever a function
 * signature changes. flashspec checks abi_version() at import time and
 * falls back to the pure-Python path on mismatch rather than risking
 * silent wrong results.
 *
 * License: Apache 2.0
 */

#ifndef FLASHSPEC_ASM_H
#define FLASHSPEC_ASM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * ABI version
 * ------------------------------------------------------------------------- */
#define FLASHSPEC_ASM_ABI_VERSION 1

/** Return the compiled-in ABI version. flashspec checks this at import. */
int abi_version(void);

/* -------------------------------------------------------------------------
 * CPU feature detection
 * ------------------------------------------------------------------------- */

/** Returns 1 if the CPU supports AVX2 and the OS has enabled YMM state. */
int cpu_supports_avx2(void);

/** Returns 1 if the CPU supports AVX-512F and the OS has enabled ZMM state.
 *  Stretch goal — always returns 0 until Phase 7. */
int cpu_supports_avx512f(void);

/* -------------------------------------------------------------------------
 * Core kernel: residual distribution
 *
 * Computes the residual probability distribution used in speculative decoding
 * draft-token rejection sampling (Leviathan et al., 2023, Algorithm 1).
 *
 *   residual[v] = max(0, p[v] - q[v])
 *   residual    = residual / max(sum(residual), 1e-9)
 *
 * Parameters
 * ----------
 * p   : target model probabilities at the rejection position, shape (V,)
 * q   : draft  model probabilities at the rejection position, shape (V,)
 * out : output buffer for the residual pmf, shape (V,)
 * V   : vocabulary size
 *
 * All arrays must be float32, contiguous, and aligned to at least 4 bytes.
 * For best AVX2 performance, 32-byte alignment is preferred (not required).
 *
 * NaN contract (see ADR 0001):
 *   vmaxps(NaN, 0.0f) returns 0.0f per Intel SDM.
 *   NaN in p or q will therefore be silently zeroed in the clamped diff,
 *   not propagated to the output. This is documented behaviour, not a bug.
 * ------------------------------------------------------------------------- */

/** Dispatch wrapper: calls AVX2 kernel if supported, else C baseline. */
void residual_dist(
    const float* restrict p,
    const float* restrict q,
    float*       restrict out,
    int64_t      V
);

/** Naive C baseline (-O3 auto-vectorised). Benchmark reference. */
void residual_dist_c(
    const float* restrict p,
    const float* restrict q,
    float*       restrict out,
    int64_t      V
);

/* Phase 3 — AVX2 kernel, defined in kernel_avx2.asm. */
void residual_dist_avx2(
    const float* p,
    const float* q,
    float*       out,
    int64_t      V
);

#ifdef __cplusplus
}
#endif

#endif /* FLASHSPEC_ASM_H */
