# FlashSpec-ASM — Build system
# Targets: all, test, bench, clean
# Requires: nasm, cc (gcc or clang), python3

CC      := cc
NASM    := nasm

CFLAGS  := -O3 -march=native -fPIC -Wall -Wextra \
            -I include/
NASMFLAGS := -f elf64 -O2

SRC_ASM  := src/kernel_avx2.asm
SRC_C    := src/shim.c src/cpuid.c
HEADER   := include/flashspec_asm.h

OBJ_ASM  := build/kernel_avx2.o
OBJ_C    := $(patsubst src/%.c, build/%.o, $(SRC_C))

LIB      := libflashspec_asm.so

.PHONY: all test bench clean

all: $(LIB)

build/:
	mkdir -p build/

$(OBJ_ASM): $(SRC_ASM) | build/
	$(NASM) $(NASMFLAGS) $< -o $@

build/%.o: src/%.c $(HEADER) | build/
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(OBJ_ASM) $(OBJ_C)
	$(CC) -shared -o $@ $^

test:
	python3 -m pytest tests/ -v --tb=short

bench:
	python3 bench/bench_harness.py

profile:
	python3 bench/profile_phase0.py

clean:
	rm -rf build/ $(LIB)
