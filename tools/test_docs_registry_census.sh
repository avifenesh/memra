#!/usr/bin/env bash
# Teeth for tools/flags-table-census.py (memra #22), tools/kernels-symbol-census.py and their
# wiring.
#
# Every arm forces the outcome it asserts: a fixture that only ever watches the real registry
# pass is not evidence the check can fail. Table arms: the real registry is green; an unescaped
# pipe inside a backticked cell is red and the message names the line; a three-cell row in a
# two-column table is red; an escaped pipe is green; pipes inside a fenced code block are
# ignored; a file with no table refuses (exit 2) rather than passing vacuously. Symbol arms: the
# real tree and docs/KERNELS.md are green; a kernel that only a wildcard covers is red and
# named; a macro-stamped kernel and a header kernel the doc omits are red and named; the same
# tree with every symbol named (a template as `name<...>`) is green; an empty cu dir refuses
# (exit 2). The wiring arms prove both censuses have a caller in tools/docs-registry-census.sh,
# that the docs-registry merge guard pins the symbol census, that docs-registry-census.sh has a
# caller in ci.yml, and that this fixture itself is reached from a step ci.yml already runs
# (tools/test_check_flags.sh chains into it). Wiring is checked on comment-stripped text, so
# this comment cannot satisfy it.
#
# CPU only, no network, no cargo. Throwaway files under mktemp.
set -uo pipefail
cd -- "$(dirname -- "$0")/.."
CENSUS=tools/flags-table-census.py
SYMBOLS=tools/kernels-symbol-census.py
[[ -x "$CENSUS" ]] || { echo "test_docs_registry_census: missing or non-executable $CENSUS" >&2; exit 2; }
[[ -x "$SYMBOLS" ]] || { echo "test_docs_registry_census: missing or non-executable $SYMBOLS" >&2; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

pass=0; fail=0
ok()   { pass=$((pass+1)); echo "  ok   $1"; }
bad()  { fail=$((fail+1)); echo "  FAIL $1" >&2; }

# --- arm 1: the real registry is green -------------------------------------------------------
if out=$(python3 "$CENSUS" docs/FLAGS.md 2>&1); then
    ok "real docs/FLAGS.md: $out"
else
    bad "real docs/FLAGS.md should be green; got: $out"
fi

# --- arm 2: unescaped pipe inside a backticked cell -> red, line named ------------------------
cat > "$tmp/unescaped.md" <<'EOF'
# fixture

| flag | default | what it does |
|---|---|---|
| `MEMRA_OK` | off | a clean row |
| `MEMRA_BAD` | off | logs `route=spec|plain` per request |
EOF
rc=0; out=$(python3 "$CENSUS" "$tmp/unescaped.md" 2>&1) || rc=$?
if (( rc == 1 )) && grep -q 'unescaped.md:6: 4 cells' <<<"$out"; then
    ok "unescaped pipe is red and names line 6"
else
    bad "unescaped pipe: rc=$rc out=$out"
fi

# --- arm 3: three-cell row in a two-column table -> red ------------------------------------
cat > "$tmp/twocol.md" <<'EOF'
| flag | what it does |
|---|---|
| `MEMRA_A` | fine |
| `MEMRA_B` | on | a default cell that has no header |
EOF
rc=0; out=$(python3 "$CENSUS" "$tmp/twocol.md" 2>&1) || rc=$?
if (( rc == 1 )) && grep -q 'twocol.md:4: 3 cells, but the table header at line 1 has 2' <<<"$out"; then
    ok "three cells in a two-column table is red"
else
    bad "two-column overflow: rc=$rc out=$out"
fi

# --- arm 4: escaped pipe -> green ----------------------------------------------------------
cat > "$tmp/escaped.md" <<'EOF'
| flag | default | what it does |
|---|---|---|
| `MEMRA_OK` | off | logs `route=spec\|plain` per request |
EOF
if out=$(python3 "$CENSUS" "$tmp/escaped.md" 2>&1) && grep -q 'rows=1' <<<"$out"; then
    ok "escaped pipe is green"
else
    bad "escaped pipe should be green; got: $out"
fi

# --- arm 5: pipes inside a fenced block are not table rows -----------------------------------
cat > "$tmp/fenced.md" <<'EOF'
| flag | default | what it does |
|---|---|---|
| `MEMRA_OK` | off | fine |

```
| this | is | a | log | line | not | a | row |
```
EOF
if out=$(python3 "$CENSUS" "$tmp/fenced.md" 2>&1) && grep -q 'tables=1 rows=1' <<<"$out"; then
    ok "fenced pipes are ignored"
else
    bad "fenced block: got: $out"
fi

# --- arm 6: no table at all -> refuses (exit 2), never a vacuous pass ----------------------
printf '# nothing here\n\nprose only\n' > "$tmp/empty.md"
rc=0; out=$(python3 "$CENSUS" "$tmp/empty.md" 2>&1) || rc=$?
if (( rc == 2 )); then
    ok "a registry with no table refuses (exit 2)"
else
    bad "no-table input should exit 2; rc=$rc out=$out"
fi

# --- arm 7: the real cu/ tree is fully named in docs/KERNELS.md --------------------------------
if out=$(python3 "$SYMBOLS" 2>&1) && grep -q 'all named in KERNELS.md' <<<"$out"; then
    ok "real KERNELS.md names every entry symbol: $out"
else
    bad "real KERNELS.md should name every entry symbol; got: $out"
fi

# A small TU with every extraction shape: extern "C" kernels, a template kernel behind a host
# launcher, extern "C" host functions on their own line and in a block (the static helper in the
# block is not an entry), a macro that stamps kernels with ##, and a cu/ header it includes.
mkdir -p "$tmp/cu"
cat > "$tmp/cu/fix.cu" <<'EOF'
#include "fix_tail.cuh"
extern "C" __global__ void fix_add_f32(float* y, const float* x) { y[0] += x[0]; }
extern "C" __global__ void fix_add_rows_f32(float* y, const float* x) { y[1] += x[1]; }
template <int N>
static __global__ void fix_tile_kernel(float* y) { y[N] = 0.0f; }
extern "C" int memra_fix_launch(float* y) { fix_tile_kernel<4><<<1, 32>>>(y); return 0; }
extern "C" {
static int fix_helper(int n) { return n; }
int memra_fix_bytes(int n) { return fix_helper(n) * 4; }
}
#define FIX_WRAP(W) \
extern "C" __global__ void fix_wide_b##W##_f32(float* y) { y[W] = 1.0f; }
FIX_WRAP(2)
FIX_WRAP(8)
#undef FIX_WRAP
EOF
cat > "$tmp/cu/fix_tail.cuh" <<'EOF'
__global__ void fix_tail_kernel(float* y) { y[2] = 0.0f; }
extern "C" int memra_fix_tail(float* y) { fix_tail_kernel<<<1, 1>>>(y); return 0; }
EOF
named='`fix_add_f32`, `fix_add_rows_f32`, `memra_fix_launch`, `memra_fix_bytes`, `fix_tile_kernel<...>`, `memra_fix_tail`'

# --- arm 8: a kernel that only a wildcard covers is red, by name -------------------------------
printf '| Symbol | Computes |\n| --- | --- |\n| `fix_add_*`, %s, `fix_wide_b2_f32`, `fix_wide_b8_f32`, `fix_tail_kernel` | fixture |\n' \
    '`fix_add_f32`, `memra_fix_launch`, `memra_fix_bytes`, `fix_tile_kernel<...>`, `memra_fix_tail`' > "$tmp/wild.md"
rc=0; out=$(python3 "$SYMBOLS" --cu-dir "$tmp/cu" --doc "$tmp/wild.md" 2>&1) || rc=$?
if (( rc == 1 )) && grep -q 'not named: fix_add_rows_f32 (cu/fix.cu)' <<<"$out" \
        && grep -q '1 of 9 entry symbols' <<<"$out"; then
    ok "a kernel only a wildcard covers is red and named"
else
    bad "wildcard-only kernel: rc=$rc out=$out"
fi

# --- arm 9: a macro-stamped kernel and a header kernel the doc omits are red, by name ----------
printf '| Symbol | Computes |\n| --- | --- |\n| %s, `fix_wide_b2_f32` | fixture |\n' "$named" > "$tmp/omit.md"
rc=0; out=$(python3 "$SYMBOLS" --cu-dir "$tmp/cu" --doc "$tmp/omit.md" 2>&1) || rc=$?
if (( rc == 1 )) && grep -q 'not named: fix_wide_b8_f32 (cu/fix.cu)' <<<"$out" \
        && grep -q 'not named: fix_tail_kernel (cu/fix.cu)' <<<"$out" \
        && grep -q '2 of 9 entry symbols' <<<"$out"; then
    ok "an omitted macro-stamped kernel and header kernel are red and named"
else
    bad "macro/header omission: rc=$rc out=$out"
fi

# --- arm 10: every symbol named exactly is green (and the static helper is not an entry) -------
printf '| Symbol | Computes |\n| --- | --- |\n| %s, `fix_wide_b2_f32`, `fix_wide_b8_f32`, `fix_tail_kernel` | fixture |\n' \
    "$named" > "$tmp/full.md"
if out=$(python3 "$SYMBOLS" --cu-dir "$tmp/cu" --doc "$tmp/full.md" 2>&1) \
        && grep -q '9 entry symbols in 1 TUs (6 kernels, 3 host entries), all named' <<<"$out"; then
    ok "a doc naming every fixture symbol is green"
else
    bad "fully named fixture should be green; got: $out"
fi

# --- arm 11: an empty cu dir refuses (exit 2), never a vacuous pass ----------------------------
mkdir -p "$tmp/nocu"
rc=0; out=$(python3 "$SYMBOLS" --cu-dir "$tmp/nocu" --doc "$tmp/full.md" 2>&1) || rc=$?
if (( rc == 2 )); then
    ok "an empty cu dir refuses (exit 2)"
else
    bad "empty cu dir should exit 2; rc=$rc out=$out"
fi

# --- wiring arms (comment-stripped text) -----------------------------------------------------
# Capture, then grep: `producer | grep -q` under pipefail races the producer SIGPIPE and reads
# a real match as a miss on any file longer than one pipe buffer (ci.yml is one).
strip() { sed -e 's/^[[:space:]]*#.*$//' "$1"; }
census_text=$(strip tools/docs-registry-census.sh)
ci_text=$(strip .github/workflows/ci.yml)
flags_fixture_text=$(strip tools/test_check_flags.sh)
if grep -q 'flags-table-census.py' <<<"$census_text"; then
    ok "docs-registry-census.sh invokes the table census"
else
    bad "docs-registry-census.sh does not invoke flags-table-census.py"
fi
if grep -q 'tools/kernels-symbol-census.py' <<<"$census_text"; then
    ok "docs-registry-census.sh invokes the symbol census"
else
    bad "docs-registry-census.sh does not invoke kernels-symbol-census.py"
fi
guard_inputs=$(python3 -c 'import json, sys
rows = json.load(open(sys.argv[1]))["guards"]
print("\n".join(next(r["inputs"] for r in rows if r["id"] == "docs-registry")))' tools/ci_merge_validation.json 2>&1 || true)
if grep -qx 'tools/kernels-symbol-census.py' <<<"$guard_inputs"; then
    ok "the docs-registry merge guard pins the symbol census"
else
    bad "tools/ci_merge_validation.json docs-registry inputs lack tools/kernels-symbol-census.py: $guard_inputs"
fi
if grep -q 'tools/docs-registry-census.sh' <<<"$ci_text"; then
    ok "ci.yml runs docs-registry-census.sh"
else
    bad "ci.yml does not run docs-registry-census.sh"
fi
if grep -q 'tools/test_docs_registry_census.sh' <<<"$flags_fixture_text"; then
    ok "test_check_flags.sh chains into this fixture (its ci.yml step is the caller)"
else
    bad "this fixture has no caller: test_check_flags.sh does not chain into it"
fi

EXPECTED_ASSERTIONS=16
total=$((pass + fail))
printf '\ntest_docs_registry_census: %d passed, %d failed (%d assertions, expected %d)\n' \
    "$pass" "$fail" "$total" "$EXPECTED_ASSERTIONS"
if (( total != EXPECTED_ASSERTIONS )); then
    printf 'test_docs_registry_census: BROKEN FIXTURE, recorded %d assertions, expected %d\n' \
        "$total" "$EXPECTED_ASSERTIONS" >&2
    exit 3
fi
(( fail == 0 ))
