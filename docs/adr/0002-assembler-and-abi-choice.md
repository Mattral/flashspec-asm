# ADR 0002 — Assembler and ABI Choice

**Status:** Accepted  
**Phase:** Project setup

---

## Context

The project requires a choice of assembler (NASM vs. GAS vs. MASM) and calling
convention (System V AMD64 vs. Microsoft x64) before writing any code.

## Decision

**Assembler: NASM with Intel syntax**  
**ABI: System V AMD64 (Linux x86-64)**

## Rationale

**NASM + Intel syntax:**
- Intel syntax (`mov rax, 1` vs. AT&T `movq $1, %rax`) is more readable for a
  portfolio piece and matches the reference documentation (Intel SDM, Agner Fog
  optimization manuals, Intel Intrinsics Guide).
- NASM assembles cleanly to ELF object files; linking with `cc` on Linux
  requires no extra flags.
- NASM's macro system is minimal and readable — unlike GAS, it doesn't require
  `.intel_syntax noprefix` guards that surprise readers.
- Consistent with how the eventual writeup will present the code.

**System V AMD64 ABI:**
- The only target platform for v1 is Linux x86-64 (the Docker/K8s deployment
  environment already used by guardrail-rs and KANX).
- The C shim (`shim.c`) handles stack alignment to 16-byte boundary before
  calling the assembly function, per the SysV ABI requirement.
- Register arguments: `rdi, rsi, rdx, rcx, r8, r9` — all that's needed for
  the four-argument `residual_dist(p, q, out, V)` signature.
- Callee-saved registers: `rbx, rbp, r12–r15` — the AVX2 kernel will use
  ymm registers (ymm0–ymm7 are caller-saved; ymm8–ymm15 are callee-saved on
  Linux). The kernel will prefer ymm0–ymm7 to avoid save/restore overhead.
- No `vzeroupper` issue: kernel will call `vzeroupper` on exit to avoid the
  AVX↔SSE transition penalty on Intel CPUs (not needed on AMD but harmless).

## Rejected Alternatives

**GAS (GNU Assembler) with AT&T syntax:** Widely available, but AT&T syntax
is harder to read and compare against Intel SDM operand order. Not justified
for a portfolio project where readability is a first-class goal.

**MASM / NASM on Windows (Microsoft x64 ABI):** Out of scope for v1 per
the project guideline. Arguments 1–4 use `rcx, rdx, r8, r9` instead of
`rdi, rsi, rdx, rcx` — a different register layout that would require a
separate kernel entry point. Post-v1 stretch goal only.

## Build System Implications

```makefile
# Assemble NASM → ELF object
kernel_avx2.o: src/kernel_avx2.asm
    nasm -f elf64 -O2 src/kernel_avx2.asm -o kernel_avx2.o

# C shim → object
shim.o: src/shim.c include/flashspec_asm.h
    cc -O3 -march=native -fPIC -c src/shim.c -o shim.o

# Link into shared object
libflashspec_asm.so: kernel_avx2.o shim.o
    cc -shared -o libflashspec_asm.so kernel_avx2.o shim.o
```

The `.so` is the artifact distributed in the GitHub release and bundled in
the `cibuildwheel` wheel for the eventual `flashspec-asm-kernel` PyPI package.
