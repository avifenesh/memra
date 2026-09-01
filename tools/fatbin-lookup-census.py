#!/usr/bin/env python3
"""Fatbin-vs-lookup census: refuses a build whose fatbins lack a kernel the Rust side looks up.

WHY THIS EXISTS
---------------
`Engine::func` (crates/memra-engine/src/lib.rs:1418) resolves kernels LAZILY BY NAME, walking
seven fatbins and ending in `panic!("kernel {name} not in any fatbin")`. So an arch-scoped `#if`
that drops an `extern "C" __global__` with no `#else` is a RUNTIME failure on a shipped binary
that every compile and link gate passes.

That is not hypothetical. cu/hybrid.cu:1575-2238 is
`#if !defined(MEMRA_PORTABLE_CUDA) || defined(MEMRA_HOPPER_MMA)` with no `#else`, so sm_89 loses
~17 kernels — several with no MMA in them at all, excluded only because they sit inside an
MMA-scoped `#if`. The single-sequence GDN path is guarded (lib.rs:23967 on
`!portable_mma_gated()`); the batched twin at hybrid_forward.rs:3912 is not. The v0.107.0 sm_89
tarball therefore panics on first batched GDN decode — and it took closing a LINK failure to
reveal it, because until then that matrix cell died before producing a tarball at all.

WHAT THIS PROVES, AND WHAT IT DOES NOT
--------------------------------------
It reads what nvcc actually emitted (`cuobjdump --list-text`, a HOST tool — no GPU) rather than
re-evaluating the preprocessor and hoping to agree with the toolchain. It proves every looked-up
name IS PRESENT in some fatbin for this arch.

It does NOT prove the code path runs. That needs real sm_89 (Ada) and sm_90a (Hopper) silicon;
the rig is sm_120a and prod boxes are untouchable. The class is narrowed here, not closed —
docs/RELEASING.md says so where sm_89/sm_90a support is described.

REFUSALS (all fail-closed; the empty cases are refusals, not passes)
  1. no fatbins found, or a fatbin `func()` searches is missing
  2. a fatbin yields zero kernel names
  3. zero `.func(...)` literals extracted from the Rust sources
  4. a fatbin built for a different arch than the one under census
  5. a looked-up name absent from every fatbin and not declared in the exceptions file
  6. EXCEPTION DRIFT — a declared exception whose symbol is now present for that arch
     (remediated, so the grant is dead) or whose symbol no longer appears in the Rust sources
     (renamed or deleted, so the grant now excuses whatever next occupies the name)

Refusal 6 exists because of a measured precedent, not a hunch. tools/check-flags.sh has the same
grandfather-list shape (research/docsync3-20260811/flags-drift.txt, 75 names) and nothing checks
its entries are still meaningful: all 75 are documented in docs/FLAGS.md today, every exemption
is dead, and a probe confirms that DELETING a documented row for one of those names keeps the
census GREEN (exit 0, name printed under "uncovered runtime names" as a non-fatal line). A
grandfather list without a drift check silently absorbs findings it was never granted for.

Usage: tools/fatbin-lookup-census.py --arch <120a|100a|90a|89> [--out-dir DIR] [--crate DIR]
  --out-dir defaults to the newest target/*/build/memra-engine-*/out containing fatbins.
Callers: ci.yml release-arch-mirror (after the build, per arch),
         tools/test_fatbin_lookup_census.sh (teeth).
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# `SASS text section 1 : x-gdn_k2_wgmma_vl.sm_100a.elf.bin`
SECTION_RE = re.compile(r"^SASS text section \d+\s*:\s*x-(?P<name>.+)\.sm_(?P<arch>[0-9a-z]+)\.elf\.bin\s*$")
# Kernel lookups: .func("name") and .func_g("name").
LOOKUP_RE = re.compile(r'\.func(?:_g)?\("([a-z0-9_]+)"\)')
# The seven modules Engine::func walks, by fatbin stem (build.rs:251-257).
REQUIRED = ["kernels", "hybrid", "qmatvec", "flash_attn", "qmatvec_gemm", "moe_router", "spec_sample"]


def die(msg: str) -> int:
    print(f"::error::fatbin-census: {msg}")
    return 1


def fatbin_symbols(cuobjdump: str, path: Path):
    """(kernel names, arches seen) for one fatbin."""
    try:
        out = subprocess.run([cuobjdump, "--list-text", str(path)],
                             capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit(die(f"{cuobjdump} not found — this census needs the CUDA toolkit's "
                             f"cuobjdump (a host tool; no GPU required)"))
    except subprocess.CalledProcessError as e:
        raise SystemExit(die(f"cuobjdump failed on {path}: {e.stderr.strip()}"))
    names, arches = set(), set()
    for line in out.splitlines():
        m = SECTION_RE.match(line.strip())
        if m:
            names.add(m.group("name"))
            arches.add(m.group("arch"))
    return names, arches


def read_exceptions(path: Path):
    """`<arch> <symbol>  # reason` -> {(arch, symbol): reason}"""
    out = {}
    if not path.is_file():
        return out
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(die(f"{path}:{lineno}: expected '<arch> <symbol>', got {raw!r}"))
        out[(parts[0], parts[1])] = raw.split("#", 1)[1].strip() if "#" in raw else ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--out-dir")
    ap.add_argument("--crate", default="crates/memra-engine")
    # Resolved rather than pinned: the pinned toolkit path first (what CI installs), then PATH
    # (other toolkit layouts, and the dev rig where it is symlinked). Still fails closed if
    # neither exists — see fatbin_symbols. cuobjdump ships in its own package
    # (cuda-cuobjdump-13-1), NOT in cuda-nvcc, which is how the first CI run of this census
    # failed.
    ap.add_argument("--cuobjdump", default=None)
    ap.add_argument("--exceptions", default="tools/fatbin-lookup-exceptions.txt")
    ap.add_argument("--advisory", default="tools/fatbin-census-advisory.txt")
    a = ap.parse_args()

    if not a.cuobjdump:
        pinned = Path("/usr/local/cuda-13.1/bin/cuobjdump")
        a.cuobjdump = str(pinned) if pinned.is_file() else (shutil.which("cuobjdump") or str(pinned))

    crate = Path(a.crate)

    # ---- locate the fatbins -------------------------------------------------------------
    if a.out_dir:
        out_dir = Path(a.out_dir)
    else:
        cands = sorted(Path(".").glob("target/*/build/memra-engine-*/out"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        cands = [c for c in cands if list(c.glob("*.fatbin"))]
        if not cands:
            return die("no target/*/build/memra-engine-*/out directory with fatbins — build "
                       "first; refusing rather than reporting a vacuous pass")
        out_dir = cands[0]

    fatbins = sorted(out_dir.glob("*.fatbin"))
    if not fatbins:                                                          # refusal 1
        return die(f"no *.fatbin in {out_dir} — refusing rather than comparing against nothing")
    have_stems = {f.stem for f in fatbins}
    missing_mods = [m for m in REQUIRED if m not in have_stems]
    if missing_mods:                                                         # refusal 1
        return die(f"{out_dir} is missing fatbin(s) Engine::func searches: "
                   f"{', '.join(missing_mods)} — an incomplete build cannot answer this")

    # ---- the supply side ----------------------------------------------------------------
    present, arches_seen = set(), set()
    for f in fatbins:
        names, arches = fatbin_symbols(a.cuobjdump, f)
        if not names:                                                        # refusal 2
            return die(f"{f} yielded zero kernel names — cuobjdump output unparsed or the "
                       f"fatbin is empty; refusing rather than treating it as 'nothing missing'")
        present |= names
        arches_seen |= arches
    bad_arch = arches_seen - {a.arch}
    if bad_arch:                                                             # refusal 4
        return die(f"fatbins in {out_dir} contain SASS for {', '.join(sorted(bad_arch))} but "
                   f"this census was asked about sm_{a.arch} — stale build directory")

    # ---- the demand side ----------------------------------------------------------------
    looked_up = set()
    src = crate / "src"
    for rs in sorted(src.rglob("*.rs")):
        looked_up |= set(LOOKUP_RE.findall(rs.read_text()))
    if not looked_up:                                                        # refusal 3
        return die(f"zero kernel lookups found under {src} — the pattern stopped matching, so "
                   f"this census would pass vacuously")

    advisory = set()
    ap_path = Path(a.advisory)
    if ap_path.is_file():
        for raw in ap_path.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                advisory.add(line)

    exceptions = read_exceptions(Path(a.exceptions))
    arch_exc = {sym for (arch, sym) in exceptions if arch == a.arch}

    # ---- refusal 5 ----------------------------------------------------------------------
    missing = sorted(looked_up - present)
    unexcused = [s for s in missing if s not in arch_exc]
    excused = [s for s in missing if s in arch_exc]

    # ---- refusal 6: the exceptions file must not rot ------------------------------------
    drift = []
    for (arch, sym) in sorted(exceptions):
        if arch != a.arch:
            continue
        if sym in present:
            drift.append(f"{arch} {sym}: now PRESENT in the fatbins — the grant is dead, "
                         f"delete the line (a live exception silently excuses whatever next "
                         f"occupies this name)")
        elif sym not in looked_up:
            drift.append(f"{arch} {sym}: no longer looked up anywhere in {src} — renamed or "
                         f"removed, so this grant now covers nothing, delete the line")

    print(f"fatbin-census: sm_{a.arch} — {len(fatbins)} fatbins, {len(present)} kernels present, "
          f"{len(looked_up)} looked up, {len(excused)} excused, {len(unexcused)} unexcused")
    for s in excused:
        print(f"  excused (declared, sm_{a.arch}): {s}")

    # Exception rot is ALWAYS fatal, advisory arch or not: a rotted grant is a gate that has
    # stopped meaning what it says, and that is never advisory.
    if drift:                                                                # refusal 6
        return die(f"exceptions file has rotted:\n  " + "\n  ".join(drift))

    if unexcused:                                                            # refusal 5
        detail = (f"sm_{a.arch}: {len(unexcused)} looked-up kernel(s) are in NO fatbin: "
                  + ", ".join(unexcused)
                  + f" — Engine::func panics 'kernel <name> not in any fatbin' at runtime on "
                    f"this arch. Either give the kernel an #else fallback, guard the call site "
                    f"the way lib.rs:23967 guards its twin, or declare it in {a.exceptions} "
                    f"with the arch and a reason.")
        if a.arch in advisory:
            # Reported in full, non-blocking — and safe ONLY because
            # tools/arch-matrix-census.sh refuses to let an advisory arch appear in
            # release.yml's matrix, so an arch in this state cannot be shipped.
            print(f"::warning::fatbin-census: ADVISORY sm_{a.arch} — {detail}")
            print(f"fatbin-census: sm_{a.arch} ADVISORY ({len(unexcused)} missing, "
                  f"non-blocking per {a.advisory}; this arch must not appear in release.yml)")
            return 0
        return die(detail)

    print(f"fatbin-census: sm_{a.arch} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
