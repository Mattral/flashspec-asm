; FlashSpec-ASM — residual_dist AVX2 kernel
; Phase 3: hand-written NASM, Intel syntax, ELF64, System V AMD64 ABI
;
; void residual_dist_avx2(const float* p, const float* q, float* out, int64_t V)
;   rdi = p   (const float* — target probs)
;   rsi = q   (const float* — draft  probs)
;   rdx = out (float*       — output buffer)
;   rcx = V   (int64_t      — vocab size)
;
; Algorithm (two-pass):
;   Pass 1: diff[v]  = max(0, p[v] - q[v])   ; vsubps + vmaxps
;           sum     += diff[v]                ; vaddps (ymm) + scalar tail
;           out[v]   = diff[v]               ; store (un-normalised)
;   Pass 2: out[v]   = out[v] * (1 / max(sum, 1e-9))  ; vbroadcastss + vmulps
;
; Register allocation:
;   rdi, rsi, rdx, rcx  = arguments (SysV AMD64)
;   r8                  = main loop bound (V & ~7)
;   rax                 = loop index
;   ymm0                = p chunk
;   ymm1                = q chunk
;   ymm2                = diff = p - q
;   ymm3                = clamped = max(0, diff)
;   ymm4                = ymm accumulator for vectorised sum (8 lanes)
;   ymm5                = zero constant (vmaxps clamp)
;   xmm6                = scalar tail accumulator + final sum
;   ymm7                = broadcast 1/denom for pass 2
;
; Bug fix (v2): tail elements are accumulated in xmm6 (separate scalar),
; not into xmm4/ymm4. The hreduce collapses ymm4 to xmm6, then the
; scalar tail sum in xmm6 is added. This prevents double-counting the
; upper lanes of ymm4 when the tail vaddss was applied to xmm4 directly.
;
; See ADR 0001 (kernel selection), 0002 (ABI), 0003 (AVX2), 0004 (C baseline)

bits 64
default rel
section .note.GNU-stack noalloc noexec nowrite progbits

section .rodata
align 16
denom_guard:  dd 1.0e-9
one_f:        dd 1.0

section .text
global residual_dist_avx2

; ---------------------------------------------------------------------------
residual_dist_avx2:
; ---------------------------------------------------------------------------
    test    rcx, rcx
    jle     .done

    ; ymm5 = 0.0  (clamp lower bound)
    vxorps  ymm5, ymm5, ymm5

    ; ymm4 = 0.0  (vectorised sum accumulator — ymm lanes only, main loop)
    vxorps  ymm4, ymm4, ymm4

    ; xmm6 = 0.0  (scalar tail sum accumulator — kept separate from ymm4)
    vxorps  xmm6, xmm6, xmm6

    xor     eax, eax
    mov     r8, rcx
    and     r8, -8              ; r8 = V & ~7 (main loop bound)

; ---------------------------------------------------------------------------
; Pass 1 main loop — 8 floats per iteration, sum into ymm4
; ---------------------------------------------------------------------------
.pass1_main:
    cmp     rax, r8
    jge     .pass1_tail

    vmovups ymm0, [rdi + rax*4]
    vmovups ymm1, [rsi + rax*4]
    vsubps  ymm2, ymm0, ymm1
    vmaxps  ymm3, ymm2, ymm5
    vaddps  ymm4, ymm4, ymm3       ; accumulate into ymm (all 8 lanes)
    vmovups [rdx + rax*4], ymm3

    add     rax, 8
    jmp     .pass1_main

; ---------------------------------------------------------------------------
; Pass 1 tail — scalar, remainder V % 8 elements
; Accumulate into xmm6 (separate from ymm4 to avoid hreduce corruption)
; ---------------------------------------------------------------------------
.pass1_tail:
    cmp     rax, rcx
    jge     .hreduce

    vmovss  xmm0, [rdi + rax*4]
    vmovss  xmm1, [rsi + rax*4]
    vsubss  xmm2, xmm0, xmm1
    vmaxss  xmm3, xmm2, xmm5      ; xmm5 low lane = 0.0 (safe)
    vaddss  xmm6, xmm6, xmm3      ; accumulate tail into SCALAR xmm6
    vmovss  [rdx + rax*4], xmm3

    inc     rax
    jmp     .pass1_tail

; ---------------------------------------------------------------------------
; Horizontal reduction: collapse ymm4 (8 vectorised partial sums) → scalar
; then add the scalar tail sum from xmm6
;
; vhaddps operates within 128-bit lanes:
;   after step 1: [a0+a1, a2+a3, a0+a1, a2+a3 | a4+a5, a6+a7, a4+a5, a6+a7]
;   after step 2: [a0+a1+a2+a3, ...           | a4+a5+a6+a7, ...]
;   extract upper 128-bit lane, add to lower → full 8-element sum
; ---------------------------------------------------------------------------
.hreduce:
    vhaddps ymm4, ymm4, ymm4       ; pairwise within each 128-bit lane
    vhaddps ymm4, ymm4, ymm4       ; pairs again → 2 partial sums in lanes 0,4
    vextractf128 xmm0, ymm4, 1     ; xmm0 = upper 128-bit lane (a4+a5+a6+a7)
    vaddss  xmm0, xmm0, xmm4      ; xmm0 = vectorised total sum (lanes 0..7)
    vaddss  xmm6, xmm6, xmm0      ; xmm6 = vectorised sum + scalar tail sum

    ; Apply denom guard: sum = max(sum, 1e-9)
    vmovss  xmm0, [denom_guard]
    vmaxss  xmm6, xmm6, xmm0

    ; Compute 1.0 / sum  (full division — not rcp approximation)
    vmovss  xmm0, [one_f]
    vdivss  xmm0, xmm0, xmm6      ; xmm0 = 1/sum

    ; Broadcast scalar reciprocal to all 8 ymm7 lanes
    vbroadcastss ymm7, xmm0

; ---------------------------------------------------------------------------
; Pass 2 main loop — multiply stored out[] by 1/sum (8 wide)
; ---------------------------------------------------------------------------
    xor     eax, eax

.pass2_main:
    cmp     rax, r8
    jge     .pass2_tail

    vmovups ymm0, [rdx + rax*4]
    vmulps  ymm0, ymm0, ymm7
    vmovups [rdx + rax*4], ymm0

    add     rax, 8
    jmp     .pass2_main

; ---------------------------------------------------------------------------
; Pass 2 tail
; ---------------------------------------------------------------------------
.pass2_tail:
    cmp     rax, rcx
    jge     .epilogue

    vmovss  xmm0, [rdx + rax*4]
    vmulss  xmm0, xmm0, xmm7      ; xmm7 low lane = 1/sum
    vmovss  [rdx + rax*4], xmm0

    inc     rax
    jmp     .pass2_tail

; ---------------------------------------------------------------------------
.epilogue:
    vzeroupper
.done:
    ret
