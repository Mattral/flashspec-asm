# FlashSpec-ASM — Build system
# Targets: all, bench-native, test, bench, profile, clean
# Requires: nasm, cc (gcc or clang), python3
#
# Two build flavours share this Makefile:
#
#   make               Portable build (the one that ships in the wheel).
#                      C code (shim.c, cpuid.c) is compiled for baseline
#                      x86-64 (SSE2 only). The ONLY AVX code in the library
#                      is the hand-written NASM kernel, and the CPUID check
#                      in the dispatcher gates every call into it. This is
#                      what makes residual_dist_c a safe fallback on CPUs
#                      without AVX2.
#
#   make bench-native  Benchmark/dev build: C code compiled with
#                      -march=native, so residual_dist_c is the strongest
#                      auto-vectorised baseline the local compiler can emit.
#                      NEVER ship this: on an AVX/AVX-512 build host the C
#                      fallback itself contains AVX instructions and dies
#                      with SIGILL on exactly the CPUs it exists for.
#                      Equivalent to `make MARCH=native`.
#
# Changing MARCH/MTUNE/CFLAGS forces a rebuild of the C objects (see
# build/cflags.stamp), so switching flavours never links stale objects.

CC      := cc
NASM    := nasm

MARCH   ?= x86-64
MTUNE   ?= generic

CFLAGS  := -O3 -march=$(MARCH) -mtune=$(MTUNE) -fPIC -Wall -Wextra \
            -I include/
NASMFLAGS := -f elf64 -O2
# -z noexecstack: belt and braces. kernel_avx2.asm already declares
# .note.GNU-stack, but an executable-stack .so is refused by dlopen on
# glibc >= 2.41, so make the linker guarantee it.
LDFLAGS := -shared -Wl,-z,noexecstack

SRC_ASM  := src/kernel_avx2.asm
SRC_C    := src/shim.c src/cpuid.c
HEADER   := include/flashspec_asm.h

OBJ_ASM  := build/kernel_avx2.o
OBJ_C    := $(patsubst src/%.c, build/%.o, $(SRC_C))
STAMP    := build/cflags.stamp

LIB      := libflashspec_asm.so

.PHONY: all bench-native test bench profile clean FORCE

all: $(LIB)

bench-native:
	$(MAKE) MARCH=native MTUNE=native all

build/:
	mkdir -p build/

# Rewritten only when the compile flags change, so objects built with
# different flags (e.g. -march=native) are never silently reused.
$(STAMP): FORCE | build/
	@echo '$(CC) $(CFLAGS)' | cmp -s - $@ || echo '$(CC) $(CFLAGS)' > $@

$(OBJ_ASM): $(SRC_ASM) | build/
	$(NASM) $(NASMFLAGS) $< -o $@

build/%.o: src/%.c $(HEADER) $(STAMP) | build/
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJ_ASM) $(OBJ_C)
	$(CC) $(LDFLAGS) -o $@ $^
	# Copy into the package dir: that is where the loader and the wheel look.
	cp $(LIB) flashspec_asm_kernel/$(LIB)

test:
	python3 -m pytest tests/ -v --tb=short

# The harness labels the C baseline as -march=native, so build it that way.
bench: bench-native
	python3 bench/bench_harness.py

profile:
	python3 bench/profile_phase0.py

clean:
	rm -rf build/ $(LIB) flashspec_asm_kernel/$(LIB) bindings/python/$(LIB)
