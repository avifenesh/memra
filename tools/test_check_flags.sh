#!/usr/bin/env bash
# Regression tests for tools/check-flags.sh.
#
# One case per cause of the v0.94.0 blind spot. MEMRA_ALLOW_UNKNOWN_PRETOKENIZER — the flag
# that decides whether a model with an unrecognized GGUF pre-tokenizer loads at all — was
# invisible to the flags gate for TWO independent reasons, and either one alone was enough to
# hide it:
#
#   cause 1  its crate (crates/memra-tokenizer/src) was not in the hand-written runtime_dirs
#   cause 2  the read is const-indirected: `std::env::var(ALLOW_UNKNOWN_PRETOKENIZER_ENV)`
#            carries no MEMRA_ name on its own line
#
# A gate whose entire job is catching undocumented operator flags missed the most consequential
# kind of flag there is, and the flags-docs lane's "546 census, 0 stale" claim was measured
# through the same hole. These tests fail if either cause comes back.
#
# Each case builds a throwaway fixture repo, drops the REAL script into its tools/, and runs
# it — no reimplementation of the census here, or the test would pass while the gate rots.
set -uo pipefail

cd -- "$(dirname -- "$0")/.."
SCRIPT=$PWD/tools/check-flags.sh
[[ -x "$SCRIPT" ]] || { echo "test_check_flags: missing $SCRIPT" >&2; exit 2; }
command -v rg >/dev/null || { echo "test_check_flags: rg is required" >&2; exit 2; }
# TEETH HARNESS ONLY (tools/test_gate_integrity_r2.sh, the same shape round 1 used for
# MEMRA_PUSH_RANGE). Pointing this at a census that CANNOT SEE a flag is how the live block below
# is proven able to fail; a fixture nobody has watched go red is not evidence. It affects only
# the live block's own invocations — the throwaway-repo cases above always run the real
# tools/check-flags.sh, so this cannot be used to make the fixture pass.
LIVE_GATE=${MEMRA_CHECK_FLAGS:-tools/check-flags.sh}

pass=0
fail=0

fixture() {
    # fixture <crate> <lib.rs body> <documented flag or empty>
    #
    # The three crates the old runtime_dirs list named are ALWAYS created, even when the case
    # puts its flag somewhere else. That is what makes these tests decisive rather than merely
    # red: against the pre-fix script every case must fail because the flag was INVISIBLE
    # (rc=0, "no uncovered runtime names"), not because the script bailed with rc=2 on a
    # missing directory. A test that cannot tell those two apart proves nothing about the hole.
    local crate=$1 body=$2 documented=${3:-}
    local root legacy
    root=$(mktemp -d)
    mkdir -p "$root/tools" "$root/docs" "$root/crates/$crate/src"
    for legacy in memra-engine memra-server memra-kv; do
        mkdir -p "$root/crates/$legacy/src"
        [[ -f "$root/crates/$legacy/src/lib.rs" ]] || : > "$root/crates/$legacy/src/lib.rs"
    done
    cp "$SCRIPT" "$root/tools/check-flags.sh"
    printf '%s\n' "$body" > "$root/crates/$crate/src/lib.rs"
    {
        printf '# Flags\n\n'
        [[ -n "$documented" ]] && printf '| `%s` | documented in the fixture |\n' "$documented"
    } > "$root/docs/FLAGS.md"
    printf '%s' "$root"
}

run_gate() {
    # run_gate <root> -> prints combined output, returns the gate's exit code
    #
    # No MEMRA_FLAGS_DRIFT_BASELINE and no baseline file (2026-08-23). Both are gone: the gate
    # now REFUSES when either is present, so setting the env here would make every case above
    # exit 2 instead of exercising the census. The fixture used to point it at an EMPTY file,
    # which is why these cases behaved the same before and after the cut.
    local root=$1
    ( cd "$root" && tools/check-flags.sh 2>&1 )
}

check() {
    # check <label> <expected rc> <expect-substring|-> <root>
    local label=$1 want_rc=$2 want_text=$3 root=$4 out rc
    out=$(run_gate "$root"); rc=$?
    local ok=1
    (( rc == want_rc )) || ok=0
    if [[ "$want_text" != "-" ]] && [[ "$out" != *"$want_text"* ]]; then ok=0; fi
    if (( ok )); then
        printf 'ok   %s\n' "$label"; pass=$((pass+1))
    else
        printf 'FAIL %s (rc=%s want=%s)\n%s\n' "$label" "$rc" "$want_rc" "$out" >&2
        fail=$((fail+1))
    fi
    rm -rf "$root"
}

# ---------------------------------------------------------------------------
# cause 1: a crate the old three-dir runtime_dirs list did not name.
# ---------------------------------------------------------------------------
LITERAL_READ='pub fn allow() -> bool {
    std::env::var("MEMRA_FIXTURE_LOADER_ESCAPE").as_deref() == Ok("1")
}'

check "cause1: literal read outside the old three dirs is CAUGHT" 1 \
    "MEMRA_FIXTURE_LOADER_ESCAPE" \
    "$(fixture memra-tokenizer "$LITERAL_READ")"

check "cause1: same read passes once documented" 0 \
    "no uncovered runtime names" \
    "$(fixture memra-tokenizer "$LITERAL_READ" MEMRA_FIXTURE_LOADER_ESCAPE)"

# And in a crate nobody has created yet — the point of discovering the dirs rather than
# listing them is that the NEXT crate is covered on the day it appears.
check "cause1: a brand-new crate is in the census from day one" 1 \
    "MEMRA_FIXTURE_LOADER_ESCAPE" \
    "$(fixture memra-notyetinvented "$LITERAL_READ")"

# ---------------------------------------------------------------------------
# cause 2: const-indirected read. The name and the env call are on different lines.
# ---------------------------------------------------------------------------
CONST_READ='pub const FIXTURE_ENV: &str = "MEMRA_FIXTURE_CONST_INDIRECT";

pub fn allow() -> bool {
    std::env::var(FIXTURE_ENV).as_deref() == Ok("1")
}'

check "cause2: const-indirected read is CAUGHT, in a dir the old list DID name" 1 \
    "MEMRA_FIXTURE_CONST_INDIRECT" \
    "$(fixture memra-engine "$CONST_READ")"

check "cause2: const-indirected read passes once documented" 0 \
    "no uncovered runtime names" \
    "$(fixture memra-engine "$CONST_READ" MEMRA_FIXTURE_CONST_INDIRECT)"

# The real shape in crates/memra-tokenizer/src/lib.rs: `&'static str`, a module-qualified
# call site, and var_os rather than var. All three must resolve.
STATIC_READ='pub const FIXTURE_ENV: &'"'"'static str = "MEMRA_FIXTURE_CONST_INDIRECT";

pub fn allow() -> bool {
    std::env::var_os(crate::FIXTURE_ENV).is_some()
}'

check "cause2: &'static str + module-qualified + var_os all resolve" 1 \
    "MEMRA_FIXTURE_CONST_INDIRECT" \
    "$(fixture memra-engine "$STATIC_READ")"

# ---------------------------------------------------------------------------
# Negative space. Widening detection must not start inventing flags, or the gate
# becomes noise and gets bypassed — the failure mode the boundary policy documents.
# ---------------------------------------------------------------------------
DECLARED_ONLY='// A name the code documents but never reads from the environment.
pub const FIXTURE_UNUSED: &str = "MEMRA_FIXTURE_NEVER_READ";

pub fn allow() -> bool {
    // MEMRA_FIXTURE_IN_A_COMMENT is prose, not a read.
    false
}'

check "negative: a const that is never passed to an env call is NOT a flag" 0 \
    "no uncovered runtime names" \
    "$(fixture memra-engine "$DECLARED_ONLY")"

SET_ONLY='pub fn arrange() {
    unsafe { std::env::set_var("MEMRA_FIXTURE_TEST_ONLY_SETTER", "1") };
}'

check "negative: a set_var-only name is NOT a read" 0 \
    "no uncovered runtime names" \
    "$(fixture memra-engine "$SET_ONLY")"

# ---------------------------------------------------------------------------
# THE RETIRED GRANDFATHER LIST (2026-08-23). These are the arms that justify the cut, and the
# first is the whole point of the change.
#
# The gate carried a 75-name baseline whose entries were exempt: `comm -23 uncovered baseline`
# dropped them, so an UNDOCUMENTED name in that list exited 0. By the time it was measured, all
# 75 had been documented in docs/FLAGS.md anyway — every exemption dead, and every one still able
# to absorb a future regression. The probe that proved it: delete MEMRA_SPEC's FLAGS.md row and
# the census exited 0.
#
# MEMRA_SPEC is used deliberately below. It is a real, live, load-bearing flag AND it was in the
# shipped baseline, so arm 1 is the exact regression this change closes rather than a synthetic
# stand-in: what exited 0 before the cut must exit 1 after it.
# ---------------------------------------------------------------------------
BASELINED_READ='pub fn spec() -> Option<String> {
    std::env::var("MEMRA_SPEC").ok()
}'

check "grandfather: an undocumented MEMRA_SPEC now FAILS (exited 0 before the cut)" 1 \
    "MEMRA_SPEC" \
    "$(fixture memra-engine "$BASELINED_READ")"

check "grandfather: MEMRA_SPEC passes when documented, as any other flag does" 0 \
    "no uncovered runtime names" \
    "$(fixture memra-engine "$BASELINED_READ" MEMRA_SPEC)"

# Reintroducing the file must be a DELIBERATE act, not a quiet re-grant. rc=2, not 1: this is a
# refusal to answer, not a census verdict, and the two must not be confusable.
regrant_root=$(fixture memra-engine "$BASELINED_READ")
mkdir -p "$regrant_root/research/docsync3-20260811"
printf 'MEMRA_SPEC\n' > "$regrant_root/research/docsync3-20260811/flags-drift.txt"
check "grandfather: the retired baseline file REAPPEARING is refused (rc=2), not honoured" 2 \
    "retired grandfather list is back" \
    "$regrant_root"

# The env is refused rather than IGNORED. A no-op environment variable is how a caller believes
# it is grandfathering a flag when the gate stopped honouring it — the silent-success shape this
# whole arc has been about.
env_root=$(fixture memra-engine "$BASELINED_READ")
env_out=$( cd "$env_root" && MEMRA_FLAGS_DRIFT_BASELINE=/tmp/whatever tools/check-flags.sh 2>&1 )
env_rc=$?
if (( env_rc == 2 )) && [[ "$env_out" == *"MEMRA_FLAGS_DRIFT_BASELINE is set but baselines are retired"* ]]; then
    printf 'ok   %s\n' "grandfather: MEMRA_FLAGS_DRIFT_BASELINE is refused, not silently ignored"
    pass=$((pass+1))
else
    printf 'FAIL %s (rc=%s)\n%s\n' \
        "grandfather: MEMRA_FLAGS_DRIFT_BASELINE should be refused" "$env_rc" "$env_out" >&2
    fail=$((fail+1))
fi
rm -rf "$env_root"

# ---------------------------------------------------------------------------
# The live tree: the flags that motivated all of this must be IN THE CENSUS, not merely
# mentioned in a doc.
#
# GATE-INTEGRITY-20260819 A-9. This block used to read:
#
#   if rg -q -N "env::var(_os)?[[:space:]]*\(|(option_)?env![[:space:]]*\(" crates --glob '*.rs' \
#       && rg -q "$want" docs/FLAGS.md
#
# The first conjunct never mentions $want. It asks "does ANY env::var( exist anywhere under
# crates/" — unconditionally true in this repo, and true in any repo that reads one environment
# variable. So the live assertion collapsed to "the name appears in docs/FLAGS.md", and
# MEMRA_ALLOW_UNKNOWN_PRETOKENIZER satisfies that while being invisible to the census: it was
# documented AND undetected, which is the precise state v0.94.0 shipped in. A test whose
# strongest conjunct is `true` is not a weak test, it is a different test.
#
# Three independent assertions per flag now, and each one can fail on its own:
#   census      — `check-flags.sh --list` (the gate's OWN census, not a reimplementation)
#                 contains the name. This is the conjunct that was missing.
#   documented  — the name resolves against docs/FLAGS.md, via the gate's own prefix rules.
#   not-uncovered — the name is absent from the gate's uncovered list. Distinct from
#                 `documented`: a prefix row can cover a name the gate still reports.
# ---------------------------------------------------------------------------
live_out=$("$LIVE_GATE" 2>&1); live_rc=$?
census=$("$LIVE_GATE" --list 2>&1); census_rc=$?
if (( census_rc != 0 )); then
    printf 'FAIL live: %s --list failed (rc=%s)\n%s\n' "$LIVE_GATE" "$census_rc" "$census" >&2
    fail=$((fail+1))
    census=""
fi
# The uncovered block is the gate's own report of names it saw but could not resolve.
uncovered=$(printf '%s\n' "$live_out" \
    | sed -n '/^check-flags: uncovered runtime names/,/^check-flags: [^ ]/p' \
    | rg -o 'MEMRA_[A-Z0-9_]+' || true)

for want in MEMRA_ALLOW_UNKNOWN_PRETOKENIZER MEMRA_FATBIN MEMRA_MOE_MMAP_ADVICE; do
    if printf '%s\n' "$census" | rg -qx -- "$want"; then
        printf 'ok   live: %s is IN THE CENSUS\n' "$want"; pass=$((pass+1))
    else
        printf 'FAIL live: %s is NOT in the census — the gate cannot see this read\n' "$want" >&2
        fail=$((fail+1))
    fi
    if rg -q -- "$want" docs/FLAGS.md; then
        printf 'ok   live: %s is documented\n' "$want"; pass=$((pass+1))
    else
        printf 'FAIL live: %s undocumented\n' "$want" >&2; fail=$((fail+1))
    fi
    if printf '%s\n' "$uncovered" | rg -qx -- "$want"; then
        printf 'FAIL live: %s is in the gate UNCOVERED list\n' "$want" >&2; fail=$((fail+1))
    else
        printf 'ok   live: %s is not reported uncovered\n' "$want"; pass=$((pass+1))
    fi
done

# The census must not be VACUOUS. A gate that sees nothing reports nothing uncovered and exits
# 0 — the failure mode check-flags.sh guards with its own "no crates/*/src found" refusal, and
# the one this fixture would otherwise sail through with three green membership checks against
# a list that happened to contain them.
census_n=$(printf '%s\n' "$census" | rg -c '^MEMRA_[A-Z0-9_]+$' || true)
if (( ${census_n:-0} >= 100 )); then
    printf 'ok   live: census is %s names (non-vacuous)\n' "$census_n"; pass=$((pass+1))
else
    printf 'FAIL live: census is only %s names — expected >= 100 in this tree\n' \
        "${census_n:-0}" >&2
    fail=$((fail+1))
fi

if (( live_rc == 0 )); then
    printf 'ok   live: gate is green (%s)\n' "$(printf '%s' "$live_out" | head -1)"
    pass=$((pass+1))
else
    printf 'FAIL live: gate is red\n%s\n' "$live_out" >&2; fail=$((fail+1))
fi

# The retired baseline must STAY gone in this tree. The throwaway arms above prove the gate
# refuses a reappearance; this proves nobody has restored it here — the difference between "the
# refusal works" and "there is nothing to refuse".
if [[ -e research/docsync3-20260811/flags-drift.txt ]]; then
    printf 'FAIL live: the retired grandfather list is back in this tree\n' >&2
    fail=$((fail+1))
else
    printf 'ok   live: no grandfather list in this tree (deleted 2026-08-23)\n'
    pass=$((pass+1))
fi

# ---------------------------------------------------------------------------
# The fixture's own floor (round 1's lesson, GATE-INTEGRITY-20260819 §3): a run that records
# FEWER assertions than it is supposed to is a BROKEN fixture, not a green one. Round 1's first
# draft printed "1 passed / 0 failed" while an arm was visibly failing, because subshell
# counters were discarded. Here the counters are in the main shell, but an early `return`, a
# `continue` added to the loop, or a case block deleted in a refactor all silently shrink the
# count — and the summary line would still say "0 failed".
#
# 8 original `check` cases + 4 grandfather-retirement arms + 3 flags x 3 live assertions
# + census size + gate-green + baseline-still-absent = 24.
# (This constant caught its own first draft: it was written 23 and the run refused at 19. It
# went 19 -> 24 on 2026-08-23 when the grandfather list was deleted.)
EXPECTED_ASSERTIONS=24
total=$((pass + fail))
printf '\ntest_check_flags: %d passed, %d failed (%d assertions, expected %d)\n' \
    "$pass" "$fail" "$total" "$EXPECTED_ASSERTIONS"
if (( total != EXPECTED_ASSERTIONS )); then
    printf 'test_check_flags: BROKEN FIXTURE — recorded %d assertions, expected %d\n' \
        "$total" "$EXPECTED_ASSERTIONS" >&2
    exit 3
fi
(( fail == 0 ))
