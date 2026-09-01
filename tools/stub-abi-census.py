#!/usr/bin/env python3
"""Stub-ABI census: refuses a fail-closed stub .cu that has drifted from the file it mirrors.

WHY THIS EXISTS
---------------
crates/memra-engine/build.rs substitutes fail-closed stub translation units for three real
ones on the non-sm_120a arches (see the `compile_src` chain):

    cu/mmq_fp4.cu          -> cu/mmq_fp4_stub.cu          (cuda_arch != 120a)
    cu/mmq_nvfp4_w4a8.cu   -> cu/mmq_nvfp4_w4a8_stub.cu   (portable: 89, 90a)
    cu/mmq_fp8_blk.cu      -> cu/mmq_fp8_blk_stub.cu      (portable: 89, 90a)

src/mmq_ffi.rs declares and calls that whole C surface unconditionally — the arch is a
build-time env var, so no #[cfg] can gate the Rust side. A stub is therefore an ABI MIRROR: if
the real file exports a symbol the stub does not, the arch does not LINK.

On 2026-08-22 `58ce746ad3` added memra_mmq_fp8_blk_quantize_act and memra_mmq_fp8_blk_grouped
to cu/mmq_fp8_blk.cu and not to the stub. main became unreleasable that morning. Two tags died
of it — v0.105.0 (run 32581590491) and v0.106.0 (run 32608092743) — both with
`rust-lld: error: undefined symbol` in the sm_89 matrix cell, because ci.yml compiled sm_120a
only and the substituting arches were compiled for the first time by the tag itself.

This census is text-only: no nvcc, no GPU, no cargo. It reads the REAL .cu files and the REAL
build.rs, not fixtures — the lesson of that same night, when five green release-guard fixture
arms coexisted with an unreleasable main.

REFUSALS (all fail-closed)
  1. a stub file whose real twin does not exist
  2. a symbol defined in the real file, NOT defined in the stub, AND referenced from Rust
     <- the defect class, and exactly the condition that breaks the link
  3. a stub file that build.rs never substitutes (dead mirror: it will silently rot)
  4. a stub filename referenced by build.rs that does not exist on disk

Refusal 2 is scoped to symbols Rust actually references ON PURPOSE. cu/mmq_fp8_blk.cu also
exports `memra_fp8_blk_nsm`, which appears nowhere in src/**.rs — the linker never asks for it,
so an absent stub body cannot break anything, and refusing it would be a false positive. A gate
that cries wolf gets bypassed, and a bypass habit is worse damage than the bug (the
MEMRA_SKIP_PERF_CI lesson). Unreferenced drift is reported as informational instead. The demand
set is every `memra_*` identifier appearing anywhere in the crate's Rust sources, which
over-approximates rather than under-approximates: no false negatives.

Usage: tools/stub-abi-census.py [crate_dir]
  default crate_dir = crates/memra-engine
Callers: ci.yml (every push — the one that matters), release.yml guard job, publish.yml,
         tools/test_stub_abi_census.sh (teeth).
"""

import re
import sys
from pathlib import Path

# A C definition at column 0, optionally prefixed with `extern "C"`, whose name is memra_*.
# Captures the name. Deliberately anchored at column 0: function BODIES are indented in these
# files, so this cannot mistake a call site for a definition.
DEF_RE = re.compile(
    r'^(?:extern\s+"C"\s+)?'          # optional per-function extern "C" (the stub style)
    r'(?:static\s+|inline\s+)*'
    r'[A-Za-z_][A-Za-z0-9_]*'          # return type
    r'[\s*&]+'
    r'(memra_[A-Za-z0-9_]+)\s*\('      # name
)


def defined_symbols(path: Path) -> set:
    """Symbols DEFINED (with a body) in this translation unit.

    A prototype is not an export: the stub declares `memra_mmq_nvfp4_f8f4_act_bytes` so it can
    delegate to it, and counting that as an export would let a stub look complete while
    defining nothing. So after the signature we scan forward for the first '{' or ';' and keep
    the symbol only when '{' wins.
    """
    text = path.read_text()
    lines = text.splitlines()
    out = set()
    for i, line in enumerate(lines):
        m = DEF_RE.match(line)
        if not m:
            continue
        rest = "\n".join(lines[i:i + 40])
        rest = rest[m.end(1):]
        brace, semi = rest.find("{"), rest.find(";")
        if brace != -1 and (semi == -1 or brace < semi):
            out.add(m.group(1))
    return out


def rust_demand(crate: Path) -> set:
    """Every memra_* identifier appearing in the crate's Rust sources.

    These are the symbols the linker can be asked for. Over-approximates (a name in a comment
    counts) — deliberate: a false refusal here is cheap to fix, a missed one costs a release.
    """
    want = set()
    src = crate / "src"
    if src.is_dir():
        for rs in src.rglob("*.rs"):
            want |= set(re.findall(r"\b(memra_[A-Za-z0-9_]+)\b", rs.read_text()))
    return want


def main() -> int:
    crate = Path(sys.argv[1] if len(sys.argv) > 1 else "crates/memra-engine")
    cu = crate / "cu"
    build_rs = crate / "build.rs"
    errors = []
    demand = rust_demand(crate)

    if not cu.is_dir():
        print(f"::error::stub-abi-census: no cu/ directory at {cu}")
        return 1
    if not build_rs.is_file():
        print(f"::error::stub-abi-census: no build.rs at {build_rs}")
        return 1

    build_src = build_rs.read_text()
    stubs = sorted(cu.glob("*_stub.cu"))
    if not stubs:
        print(f"::error::stub-abi-census: no *_stub.cu found in {cu} — the substitution "
              f"mechanism in build.rs implies at least one; refusing rather than reporting "
              f"a vacuous pass")
        return 1

    # Refusal 4: every stub build.rs names must exist.
    for name in sorted(set(re.findall(r'"(cu/[A-Za-z0-9_]+_stub\.cu)"', build_src))):
        if not (crate / name).is_file():
            errors.append(f"build.rs substitutes {name}, which does not exist on disk")

    checked = 0
    for stub in stubs:
        real = stub.with_name(stub.name.replace("_stub.cu", ".cu"))
        rel_stub = f"cu/{stub.name}"

        # Refusal 1
        if not real.is_file():
            errors.append(
                f"{rel_stub} has no real twin at cu/{real.name} — a mirror of nothing")
            continue

        # Refusal 3
        if f'"{rel_stub}"' not in build_src:
            errors.append(
                f"{rel_stub} exists but build.rs never substitutes it — a stub no arch "
                f"compiles is a mirror that rots silently; wire it or delete it")

        # Refusal 2 — the defect class, scoped to what the linker can actually demand.
        want, have = defined_symbols(real), defined_symbols(stub)
        missing = sorted(want - have)
        breaking = [s for s in missing if s in demand]
        inert = [s for s in missing if s not in demand]
        if breaking:
            errors.append(
                f"cu/{real.name} defines {len(breaking)} Rust-referenced symbol(s) that "
                f"{rel_stub} does not: " + ", ".join(breaking)
                + f" — the substituting arches will fail to LINK with 'undefined symbol'. "
                  f"Add fail-closed bodies to {rel_stub} (return a distinct refusal code; "
                  f"never make one silently succeed).")
        extra = sorted(have - want)
        checked += 1
        status = "DRIFTED" if breaking else "ok"
        print(f"stub-abi-census: {rel_stub} vs cu/{real.name}: {status} "
              f"({len(want)} exported, {len(breaking)} missing-and-referenced"
              + (f", {len(inert)} missing-but-unreferenced: {', '.join(inert)}" if inert else "")
              + (f", {len(extra)} stub-only: {', '.join(extra)}" if extra else "")
              + ")")

    if errors:
        for e in errors:
            print(f"::error::stub-abi-census: {e}")
        return 1

    print(f"stub-abi-census: OK — {checked} stub pair(s), every real export mirrored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
