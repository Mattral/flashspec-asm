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
#  include <cpuid.h>   /* __get_cpuid / __get_cpuid_max / __cpuid_count */
#endif

#if defined(__GNUC__) || defined(__clang__)
/* -------------------------------------------------------------------------
 * Read XCR0 (the OS-enabled extended-state mask) with XGETBV, ECX = 0.
 *
 * XGETBV returns the 64-bit XCR0 split across two registers: low half in
 * EAX, high half in EDX. Read both explicitly and combine them.
 *
 * Do NOT write `__asm__("xgetbv" : "=A"(xcr0) : "c"(0))`. The "A"
 * constraint means the EDX:EAX register pair only on 32-bit x86. On x86-64
 * a 64-bit operand fits in one register, so GCC and Clang allocate it to
 * RAX *or* RDX, not the pair. Both compilers pick RAX: xcr0 then holds
 * EAX zero-extended (XGETBV clears the upper halves of RAX/RDX) and the
 * high 32 bits of XCR0, which XGETBV left in EDX, are silently dropped.
 * The old code only worked because every bit tested here (1, 2, 5, 6, 7)
 * lives in the low half; it was wrong by construction and would have
 * misreported the first XCR0 check of a bit >= 32.
 *
 * Callers execute this only after CPUID has reported OSXSAVE; XGETBV raises
 * #UD otherwise.
 * ------------------------------------------------------------------------- */
static uint64_t read_xcr0(void) {
    uint32_t lo = 0, hi = 0;
    __asm__ volatile ("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((uint64_t)hi << 32) | lo;
}
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
    uint64_t xcr0 = read_xcr0();
    if ((xcr0 & 0x6u) != 0x6u)
        return 0;

    /* CPUID.07H.00H:EBX[5] = AVX2.
     * Check leaf 7 exists first: CPUID with a leaf above the max basic leaf
     * returns the data of the highest leaf instead, i.e. a bogus EBX.
     * (__get_cpuid_max/__cpuid_count rather than __get_cpuid_count, which
     * only exists from GCC 7.) */
    if (__get_cpuid_max(0, 0) < 7)
        return 0;
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

    uint64_t xcr0 = read_xcr0();
    /* Bits 1,2 (SSE/AVX) + bits 5,6,7 (opmask, ZMM_Hi256, Hi16_ZMM).
     * All three AVX-512 state components must be OS-enabled (Intel SDM
     * Vol. 1, "Detection of AVX-512 Foundation Instructions"); bit 7
     * covers zmm16-31. */
    if ((xcr0 & 0xE6u) != 0xE6u)
        return 0;

    if (__get_cpuid_max(0, 0) < 7)
        return 0;
    uint32_t eax_7 = 0, ebx_7 = 0, ecx_7 = 0, edx_7 = 0;
    __cpuid_count(7, 0, eax_7, ebx_7, ecx_7, edx_7);
    return (ebx_7 >> 16) & 1;
#else
    return 0;
#endif
}
