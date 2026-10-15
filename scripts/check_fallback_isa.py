#!/usr/bin/env python3
"""Fail if the non-AVX2 code paths of libflashspec_asm.so contain AVX code.

residual_dist() runs cpu_supports_avx2() and, on CPUs without AVX2, falls
back to residual_dist_c(). Those functions (and abi_version /
cpu_supports_avx512f, which callers run before deciding anything) execute on
CPUs that may lack AVX, so they must not contain a single VEX- or
EVEX-encoded instruction. A build with -march=native on an AVX/AVX-512 host
silently violates this: the compiler auto-vectorises residual_dist_c with
ymm/zmm registers and the "fallback" dies with SIGILL exactly where it is
needed.

Usage:
    scripts/check_fallback_isa.py path/to/libflashspec_asm.so
    scripts/check_fallback_isa.py path/to/flashspec_asm_kernel-*.whl

Flags, per checked function (compiler clones such as `.cold` and
`.part.0` included): any ymm/zmm register, any AVX-512 opmask register
(k0-k7), and any instruction whose mnemonic is an AVX one (VEX/EVEX
encodings all have `v`-prefixed mnemonics in objdump, including
xmm-only forms like `vmovss`). As a positive control, residual_dist_avx2
must contain ymm instructions, which proves the disassembly is being read.

Not covered: non-baseline but pre-AVX extensions (SSE3-SSE4.2, POPCNT,
BMI). The Makefile's -march=x86-64 is what rules those out.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

LIB_NAME = "libflashspec_asm.so"

# Functions that run before, or instead of, the AVX2 kernel.
MUST_BE_BASELINE = (
    "residual_dist_c",
    "residual_dist",
    "cpu_supports_avx2",
    "cpu_supports_avx512f",
    "abi_version",
)
POSITIVE_CONTROL = "residual_dist_avx2"

# objdump -d function header: "0000000000001220 <residual_dist_c>:"
FUNC_RE = re.compile(r"^[0-9a-f]+ <([^>]+)>:$")
# Instruction line (--no-show-raw-insn): "    1234:\tvaddps %ymm0,..."
INSN_RE = re.compile(r"^\s*[0-9a-f]+:\s+(.*)$")
WIDE_REG_RE = re.compile(r"%(?:[yz]mm\d+|k[0-7])\b")
# Mnemonics starting with "v" that are NOT AVX (VMX/SVM, segment verify).
NON_AVX_V = {
    "verr", "verw", "vmcall", "vmlaunch", "vmresume", "vmxoff", "vmxon",
    "vmread", "vmwrite", "vmptrld", "vmptrst", "vmclear", "vmfunc",
    "vmrun", "vmload", "vmsave", "vmmcall",
}
# Prefixes objdump may print before the mnemonic.
PREFIXES = {"lock", "rep", "repe", "repz", "repne", "repnz", "bnd", "notrack",
            "data16", "addr32", "cs", "ds", "es", "ss", "fs", "gs", "rex",
            "rex.w", "{vex}", "{vex3}", "{evex}"}


def disassemble(so: Path) -> dict[str, list[str]]:
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", str(so)],
        check=True, capture_output=True, text=True,
    ).stdout
    funcs: dict[str, list[str]] = {}
    current = None
    for line in out.splitlines():
        m = FUNC_RE.match(line)
        if m:
            current = m.group(1)
            funcs[current] = []
            continue
        if current is None:
            continue
        m = INSN_RE.match(line)
        if m:
            funcs[current].append(m.group(1).strip())
    return funcs


def mnemonic(insn: str) -> str:
    for tok in insn.split():
        if tok.lower() not in PREFIXES:
            return tok.lower()
    return ""


def avx_violations(insns: list[str]) -> list[str]:
    bad = []
    for insn in insns:
        mn = mnemonic(insn)
        if WIDE_REG_RE.search(insn) or (mn.startswith("v") and mn not in NON_AVX_V):
            bad.append(insn)
    return bad


def owns(func: str, base: str) -> bool:
    """`func` is `base` or a compiler clone of it (base.cold, base.part.0...)."""
    return func == base or func.startswith(base + ".")


def check(so: Path) -> int:
    funcs = disassemble(so)
    failed = False

    for base in MUST_BE_BASELINE:
        names = sorted(f for f in funcs if owns(f, base))
        if not names:
            print(f"FAIL {base}: symbol not found in disassembly")
            failed = True
            continue
        for name in names:
            bad = avx_violations(funcs[name])
            if bad:
                failed = True
                print(f"FAIL {name}: {len(bad)} AVX instruction(s) in a "
                      f"non-AVX2 code path, e.g.:")
                for insn in bad[:5]:
                    print(f"       {insn}")
            else:
                print(f"ok   {name}: {len(funcs[name])} instructions, no AVX")

    ctrl = [i for f, insns in funcs.items() if owns(f, POSITIVE_CONTROL)
            for i in insns if "%ymm" in i]
    if ctrl:
        print(f"ok   {POSITIVE_CONTROL}: {len(ctrl)} ymm instructions "
              f"(positive control)")
    else:
        print(f"FAIL {POSITIVE_CONTROL}: no ymm instructions found; the "
              f"kernel is missing or the disassembly is not being parsed")
        failed = True

    print("FAILED" if failed else "PASSED", so)
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(argv[1])
    if target.suffix != ".whl":
        return check(target)
    with zipfile.ZipFile(target) as whl:
        members = [n for n in whl.namelist() if n.endswith("/" + LIB_NAME)]
        if len(members) != 1:
            print(f"FAIL: expected one {LIB_NAME} in {target}, found {members}")
            return 1
        with tempfile.TemporaryDirectory() as tmp:
            so = Path(tmp) / LIB_NAME
            with whl.open(members[0]) as src, open(so, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"checking {members[0]} from {target.name}")
            return check(so)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
