#!/usr/bin/env bash
# Teeth for tools/fatbin-lookup-census.py. Every refusal is FORCED, including both directions of
# exceptions-file rot, and the last arm proves ci.yml actually calls the census.
#
# No CUDA, no GPU, no network: a stub `cuobjdump` prints the same
# `SASS text section N : x-<kernel>.sm_<arch>.elf.bin` lines the real one does, reading each
# fatbin's kernel list from a sidecar `<fatbin>.syms`. That makes every arm deterministic — the
# real toolchain cannot be talked into emitting a fatbin that is missing exactly one kernel.
#
# Arm 5 reproduces the live sm_89 defect (a looked-up kernel in no fatbin) and arms 7-8 reproduce
# the grandfather-list rot measured in tools/check-flags.sh on 2026-08-23, where all 75 baselined
# names were dead and deleting a documented row still exited 0.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
census=tools/fatbin-lookup-census.py
ci=.github/workflows/ci.yml
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# Stub cuobjdump: emits sections for whatever the sidecar lists.
cat > "$tmp/cuobjdump" <<'STUB'
#!/usr/bin/env bash
# usage: cuobjdump --list-text <fatbin>
fb=${2:?}
arch=$(cat "$fb.arch" 2>/dev/null || echo 120a)
n=0
while read -r k; do
    [ -n "$k" ] || continue
    n=$((n+1))
    echo "SASS text section $n : x-$k.sm_$arch.elf.bin"
done < "$fb.syms"
STUB
chmod +x "$tmp/cuobjdump"

REQUIRED="kernels hybrid qmatvec flash_attn qmatvec_gemm moe_router spec_sample"

# stage <dir> <arch> <kernels...> : a complete, passing build tree.
stage() {
    local d=$1 arch=$2; shift 2
    rm -rf "$d"; mkdir -p "$d/out" "$d/crate/src"
    local first=1
    for m in $REQUIRED; do
        : > "$d/out/$m.fatbin"
        echo "$arch" > "$d/out/$m.fatbin.arch"
        if [ $first = 1 ]; then printf '%s\n' "$@" > "$d/out/$m.fatbin.syms"; first=0
        else printf 'filler_%s\n' "$m" > "$d/out/$m.fatbin.syms"; fi
    done
    # Rust side looks up everything staged, via both accessors.
    { echo 'fn a(){'; for k in "$@"; do echo "  self.func(\"$k\");"; done; echo '}'; } \
        > "$d/crate/src/lib.rs"
}
# Every arm passes an EMPTY advisory file unless it is testing the advisory path. Without this
# the real tools/fatbin-census-advisory.txt would leak in — it lists 89 today, which silently
# turned arm 5's expected refusal into an advisory pass. A fixture must not inherit production
# policy: it caught that immediately, which is the point.
run() { local d=$1 arch=$2; shift 2; "$census" --arch "$arch" --out-dir "$d/out" \
        --crate "$d/crate" --cuobjdump "$tmp/cuobjdump" --advisory "$tmp/no-advisory.txt" \
        "$@" 2>&1; }

: > "$tmp/none.txt"
: > "$tmp/no-advisory.txt"

# 1 — happy path must PASS, with no exceptions file needed.
stage "$tmp/a" 89 alpha_kernel beta_kernel
out=$(run "$tmp/a" 89 --exceptions "$tmp/none.txt") || fail "arm 1: refused a complete tree: $out"
echo "$out" | grep -q 'fatbin-census: sm_89 OK' || fail "arm 1: no OK line: $out"

# 2 — no fatbins at all: an empty supply side must REFUSE, never read as "nothing missing".
stage "$tmp/b" 89 alpha_kernel; rm -f "$tmp/b"/out/*.fatbin
if out=$(run "$tmp/b" 89 --exceptions "$tmp/none.txt"); then
    fail "arm 2: accepted a tree with no fatbins: $out"
fi
echo "$out" | grep -q 'no \*.fatbin' || fail "arm 2: wrong refusal: $out"

# 3 — a module Engine::func searches is absent.
stage "$tmp/c" 89 alpha_kernel; rm -f "$tmp/c/out/hybrid.fatbin"
if out=$(run "$tmp/c" 89 --exceptions "$tmp/none.txt"); then
    fail "arm 3: accepted a build missing hybrid.fatbin: $out"
fi
echo "$out" | grep -q 'hybrid' || fail "arm 3: refusal does not name the missing module: $out"

# 4 — a fatbin yields zero kernel names (parser drift or an empty fatbin).
stage "$tmp/d" 89 alpha_kernel; : > "$tmp/d/out/hybrid.fatbin.syms"
if out=$(run "$tmp/d" 89 --exceptions "$tmp/none.txt"); then
    fail "arm 4: accepted a fatbin with zero kernels: $out"
fi
echo "$out" | grep -q 'zero kernel names' || fail "arm 4: wrong refusal: $out"

# 5 — THE DEFECT CLASS: a looked-up kernel in no fatbin (the live sm_89 GDN case).
stage "$tmp/e" 89 alpha_kernel
echo '  self.func("gdn_l2_vl");' >> "$tmp/e/crate/src/lib.rs"
if out=$(run "$tmp/e" 89 --exceptions "$tmp/none.txt"); then
    fail "arm 5: accepted a lookup present in NO fatbin: $out"
fi
echo "$out" | grep -q 'gdn_l2_vl' || fail "arm 5: refusal does not name the kernel: $out"
echo "$out" | grep -q 'not in any fatbin' \
  || fail "arm 5: refusal does not name the runtime consequence: $out"

# 6 — the same tree PASSES once the absence is declared for that arch.
printf '89 gdn_l2_vl  # guarded at the call site\n' > "$tmp/exc-ok.txt"
out=$(run "$tmp/e" 89 --exceptions "$tmp/exc-ok.txt") \
  || fail "arm 6: refused a declared exception: $out"
echo "$out" | grep -q 'excused (declared, sm_89): gdn_l2_vl' \
  || fail "arm 6: excused kernel not reported: $out"

# 6b — a declaration for a DIFFERENT arch must not excuse this one.
printf '90a gdn_l2_vl  # wrong arch\n' > "$tmp/exc-wrongarch.txt"
if out=$(run "$tmp/e" 89 --exceptions "$tmp/exc-wrongarch.txt"); then
    fail "arm 6b: an exception declared for sm_90a excused sm_89: $out"
fi

# 7 — EXCEPTION ROT, direction 1: the symbol is now present, so the grant is dead.
stage "$tmp/f" 89 alpha_kernel gdn_l2_vl
if out=$(run "$tmp/f" 89 --exceptions "$tmp/exc-ok.txt"); then
    fail "arm 7: accepted a dead exception (symbol now present): $out"
fi
echo "$out" | grep -q 'now PRESENT' || fail "arm 7: wrong refusal: $out"
echo "$out" | grep -q 'delete the line' || fail "arm 7: refusal gives no remedy: $out"

# 8 — EXCEPTION ROT, direction 2: the symbol is no longer looked up (renamed/deleted), so the
#     grant now covers whatever next occupies the name. This is the arm that the flags baseline
#     lacks, and lacking it is why all 75 of its entries went dead unnoticed.
stage "$tmp/g" 89 alpha_kernel
if out=$(run "$tmp/g" 89 --exceptions "$tmp/exc-ok.txt"); then
    fail "arm 8: accepted an exception for a symbol nothing looks up: $out"
fi
echo "$out" | grep -q 'no longer looked up' || fail "arm 8: wrong refusal: $out"

# 9 — zero lookups extracted: the demand side going empty must REFUSE, not pass vacuously.
stage "$tmp/h" 89 alpha_kernel; echo 'fn nothing(){}' > "$tmp/h/crate/src/lib.rs"
if out=$(run "$tmp/h" 89 --exceptions "$tmp/none.txt"); then
    fail "arm 9: accepted a crate with zero kernel lookups: $out"
fi
echo "$out" | grep -q 'vacuously' || fail "arm 9: wrong refusal: $out"

# 10 — arch mismatch: fatbins from a stale build directory.
stage "$tmp/i" 120a alpha_kernel
if out=$(run "$tmp/i" 89 --exceptions "$tmp/none.txt"); then
    fail "arm 10: accepted sm_120a fatbins under an sm_89 census: $out"
fi
echo "$out" | grep -q 'stale build directory' || fail "arm 10: wrong refusal: $out"

# 11 — a malformed exceptions line must refuse rather than be silently skipped.
printf 'just-one-field\n' > "$tmp/exc-bad.txt"
if out=$(run "$tmp/a" 89 --exceptions "$tmp/exc-bad.txt"); then
    fail "arm 11: accepted a malformed exceptions line: $out"
fi

# 13 — ADVISORY arch: missing kernels are reported, non-blocking. Safe only because
#      tools/arch-matrix-census.sh refuses to let an advisory arch into release.yml's matrix
#      (arm 15 of tools/test_releasability_census.sh proves that half).
printf '89  # measured missing, compile-only, must not ship\n' > "$tmp/adv.txt"
out=$(run "$tmp/e" 89 --exceptions "$tmp/none.txt" --advisory "$tmp/adv.txt") \
  || fail "arm 13: advisory arch still blocked: $out"
echo "$out" | grep -q 'ADVISORY' || fail "arm 13: no ADVISORY marker: $out"
echo "$out" | grep -q 'gdn_l2_vl' \
  || fail "arm 13: advisory must still REPORT the missing kernel, not hide it: $out"
echo "$out" | grep -q 'must not appear in release.yml' \
  || fail "arm 13: advisory output does not state the shipping invariant: $out"

# 14 — advisory must NOT weaken the structural refusals. Same advisory arch, but a broken tree:
#      an empty supply side is still fatal, because "we know kernels are missing" is a different
#      statement from "we cannot tell whether kernels are missing".
stage "$tmp/j" 89 alpha_kernel; rm -f "$tmp/j"/out/*.fatbin
if out=$(run "$tmp/j" 89 --exceptions "$tmp/none.txt" --advisory "$tmp/adv.txt"); then
    fail "arm 14: advisory arch accepted a tree with no fatbins: $out"
fi

# 15 — advisory must NOT weaken exception rot either: a grant that has stopped meaning what it
#      says is never advisory.
stage "$tmp/k" 89 alpha_kernel gdn_l2_vl
if out=$(run "$tmp/k" 89 --exceptions "$tmp/exc-ok.txt" --advisory "$tmp/adv.txt"); then
    fail "arm 15: advisory arch accepted a dead exception: $out"
fi
echo "$out" | grep -q 'now PRESENT' || fail "arm 15: wrong refusal: $out"

# 16 — WIRING. Comments stripped: the prose above the step names the script too, so a plain
#      substring search would pass after the step itself was deleted.
live=$(grep -vE '^\s*#' "$ci")
echo "$live" | grep -qF 'tools/fatbin-lookup-census.py' \
  || fail "arm 16: $ci does not invoke the fatbin census"
echo "$live" | grep -qF 'tools/test_fatbin_lookup_census.sh' \
  || fail "arm 16: $ci does not run this fixture"
# It must run per-arch, inside the arch matrix — a single-arch invocation would miss the very
# arches that break.
echo "$live" | grep -qF 'matrix.cuda_arch }}' \
  || fail "arm 16: the census is not parameterised by the arch matrix"

echo "fatbin-lookup-census fixture: 16 arms PASS"
