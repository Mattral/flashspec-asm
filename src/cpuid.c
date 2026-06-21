/*
 * FlashSpec-ASM — CPUID runtime feature detection
 *
 * Provides cpu_supports_avx2() for the Phase 4 dispatch wrapper.
 * Included in v1 as a stub so the build system doesn't need restructuring
 * when AVX-512 dispatch is added in Phase 7 (stretch).
 *
 * References:
 *   Intel SDM Vol. 2A, CPUID instruction
 *   Agner Fog: Optimizing software in C++, §13.4
 */

#include "flashspec_asm.h"
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#  include <cpuid.h>   /* __get_cpuid / __cpuid_count */
#endif

/* -------------------------------------------------------------------------
 * AVX2: CPUID.07H.00H:EBX[5]
 * Also requires OS XSAVE support (CPUID.01H:ECX[27]) and
 * AVX enabled in XCR0 (checked via XGETBV).
 * ------------------------------------------------------------------------- */
int cpu_supports_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    /* Check OS XSAVE support (required for YMM state save/restore) */
    uint32_t eax_1 = 0, ebx_1 = 0, ecx_1 = 0, edx_1 = 0;
    if (!__get_cpuid(1, &eax_1, &ebx_1, &ecx_1, &edx_1))
        return 0;

    /* CPUID.01H:ECX[27] = OSXSAVE */
    if (!(ecx_1 & (1u << 27)))
        return 0;

    /* XGETBV: XCR0 bits 1 (SSE) and 2 (AVX) must be set */
    uint64_t xcr0 = 0;
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile ("xgetbv" : "=A"(xcr0) : "c"(0));
#endif
    if ((xcr0 & 0x6u) != 0x6u)
        return 0;

    /* CPUID.07H.00H:EBX[5] = AVX2 */
    uint32_t eax_7 = 0, ebx_7 = 0, ecx_7 = 0, edx_7 = 0;
    __cpuid_count(7, 0, eax_7, ebx_7, ecx_7, edx_7);
    return (ebx_7 >> 5) & 1;

#else
    /* Non-GCC/Clang: conservative fallback — assume no AVX2 */
    return 0;
#endif
}

/* -------------------------------------------------------------------------
 * AVX-512F: CPUID.07H.00H:EBX[16]
 * Stretch goal — Phase 7. Stub included now for build system stability.
 * ------------------------------------------------------------------------- */
int cpu_supports_avx512f(void) {
#if defined(__GNUC__) || defined(__clang__)
    /* Requires OSXSAVE + ZMM state in XCR0 (bits 5:6) */
    uint32_t eax_1 = 0, ebx_1 = 0, ecx_1 = 0, edx_1 = 0;
    if (!__get_cpuid(1, &eax_1, &ebx_1, &ecx_1, &edx_1))
        return 0;
    if (!(ecx_1 & (1u << 27)))
        return 0;

    uint64_t xcr0 = 0;
    __asm__ volatile ("xgetbv" : "=A"(xcr0) : "c"(0));
    /* Bits 1,2 (SSE/AVX) + bits 5,6 (opmask, ZMM_Hi256) */
    if ((xcr0 & 0x66u) != 0x66u)
        return 0;

    uint32_t eax_7 = 0, ebx_7 = 0, ecx_7 = 0, edx_7 = 0;
    __cpuid_count(7, 0, eax_7, ebx_7, ecx_7, edx_7);
    return (ebx_7 >> 16) & 1;
#else
    return 0;
#endif
}
