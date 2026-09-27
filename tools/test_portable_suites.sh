#!/usr/bin/env bash
# test_portable_suites.sh: teeth for tools/portable-suites.sh (memra #545, acceptance criterion
# 3: "a deliberately failing ownership/retirement/CLI test fails the gate; build/clippy alone
# cannot satisfy it").
#
# Same law as tools/test_flags_guard.sh and GATE-INTEGRITY-20260819 A-18: a gate nobody has
# watched fail reads as coverage while providing none. This fixture drives the REAL wrapper
# against a COPY of the tree with failures planted, so a red here can only come from the planted
# tests, and a green wrapper on the real tree cannot be the wrapper failing to run the suites.
#
# Arms:
#   1. three planted failing tests, one per crate (a retirement/ownership test in memra-tier's
#      contracts integration suite, a KV hierarchy test, an onboarding-receipt test in memra-cli)
#      -> the wrapper exits non-zero, cargo names all three targets as failed, and the banked
#      raw log lists all three planted tests under `failures:` (so --no-fail-fast holds and every
#      crate actually ran; a wrapper that stopped at the first red would name one)
#   2. a planted #[test] that prints SKIP and returns (an artifact-gated test born undeclared),
#      planted as a NEW integration-test file under crates/memra-cli/tests/ (2a) and inside
#      crates/memra-cli/src (2b) -> the wrapper exits non-zero from the STATIC skip census,
#      naming the test, before cargo runs. 2a exists because the census was src-only until
#      2026-09-21 and a tests/** skip could be born undeclared (memra #545 review).
#   3. wiring: .github/workflows/ci.yml runs the wrapper AND this fixture; tools/local-ci.sh
#      runs the wrapper (a gate outside every battery rots silently, H100 lane law 3); ONE
#      executor: ci.yml has exactly one portable-suites job (main at 34ed99dfc, 2026-09-21,
#      had two, from #592 and #590, and GitHub refused the file), no live line of ci.yml or
#      local-ci.sh runs `cargo test` on the three crates outside the wrapper, and
#      tools/ci-portable.sh (#590's name) only forwards here and runs no cargo of its own
#
# Target dir: the copy builds into target/portable-suites-teeth, NEVER the tree's own target.
# Learned on the first run (2026-09-21): cargo's metadata hash for a workspace member excludes
# its path (target dirs are relocatable), and `cp -a` keeps source mtimes, so the copy's planted
# `memra_cli` test binary landed under the same hash as the real tree's and the next real run
# reused it: the wrapper went red on the real tree with the planted test. A shared target dir is
# not a shortcut here, it is a poisoning path in both directions. The separate dir sits under
# target/ so rust-cache keeps its registry artifacts between CI runs; the workspace crates
# rebuild every time (their sources are newer than any cached output), which is the point.
# Arm 2 runs no cargo. CPU only: no GPU, no model, no network (--offline against the committed
# lockfile). Measured on the rig, 6 jobs, 2026-09-21: 27 s with a cold teeth target dir, 19 s warm.
set -uo pipefail

here=$(cd -- "$(dirname -- "$0")/.." && pwd)
wrapper=$here/tools/portable-suites.sh
[[ -x "$wrapper" ]] || { echo "test_portable_suites: missing or non-executable $wrapper" >&2; exit 2; }
bash -n "$wrapper" || { echo "test_portable_suites: $wrapper does not parse" >&2; exit 2; }

tmp=$(mktemp -d "${TMPDIR:-/tmp}/portable-suites-teeth.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

pass=0
fail=0
ok()  { printf 'ok   %s\n' "$1"; pass=$((pass+1)); }
bad() { printf 'FAIL %s\n' "$1" >&2; fail=$((fail+1)); }

stage() {
    # stage <name> -> sets the global $copy. Everything cargo and the wrapper read: the
    # workspace manifests and lockfile, the toolchain pin, .cargo (job cap), crates/ and tools/
    # as REGULAR copies (the planted edits land there; a hardlink copy would edit the tracked
    # file), research/ as a symlink (tier, bank and KV tests read committed receipts and
    # fixtures under it at test time; nothing planted touches it, and /tmp may be tmpfs).
    copy=$tmp/$1
    mkdir -p "$copy"
    cp -a "$here/Cargo.toml" "$here/Cargo.lock" "$here/rust-toolchain.toml" "$here/rustfmt.toml" "$copy/"
    [[ -d "$here/.cargo" ]] && cp -a "$here/.cargo" "$copy/.cargo"
    cp -a "$here/crates" "$copy/crates"
    cp -a "$here/tools" "$copy/tools"
    ln -s "$here/research" "$copy/research"
}

teeth_target=$here/target/portable-suites-teeth
run_wrapper() {
    # run_wrapper <log>: the real wrapper text, in the copy, in the teeth's own target dir.
    ( cd "$copy" && CARGO_TARGET_DIR=$teeth_target bash tools/portable-suites.sh ) >"$1" 2>&1
}

# ---- arm 1: planted failing tests, one per crate ------------------------------------------
stage planted
cat >> "$copy/crates/memra-tier/tests/contracts/mod.rs" <<'EOF'

#[test]
fn planted_retirement_keeps_owner_red_arm() {
    // Planted by tools/test_portable_suites.sh: a retirement that leaves its owner in place
    // must fail the gate. If this ever passes, the wrapper is not running the suite.
    assert_eq!(
        "owner-after-retire", "released",
        "planted tier retirement/ownership failure"
    );
}
EOF
cat >> "$copy/crates/memra-kv/src/lib.rs" <<'EOF'

#[cfg(test)]
mod planted_red_arm {
    #[test]
    fn planted_kv_hierarchy_red_arm() {
        panic!("planted KV hierarchy failure (tools/test_portable_suites.sh)");
    }
}
EOF
cat >> "$copy/crates/memra-cli/src/lib.rs" <<'EOF'

#[cfg(test)]
mod planted_red_arm {
    #[test]
    fn planted_onboarding_receipt_red_arm() {
        panic!("planted CLI onboarding-receipt failure (tools/test_portable_suites.sh)");
    }
}
EOF
log=$tmp/planted.log
run_wrapper "$log"
rc=$?
raw=$copy/target/portable-suites.log
if [[ $rc -ne 0 ]]; then
    ok "arm1 planted failures red the wrapper (exit $rc)"
else
    bad "arm1 wrapper exited 0 with three planted failing tests"
fi
if [[ -f "$raw" ]]; then
    ok "arm1 wrapper banked the raw cargo log"
else
    bad "arm1 no raw cargo log at $raw"
    raw=$log
fi
for planted in planted_retirement_keeps_owner_red_arm \
               planted_red_arm::planted_kv_hierarchy_red_arm \
               planted_red_arm::planted_onboarding_receipt_red_arm; do
    if grep -qE "^\s+$planted\$" "$raw"; then
        ok "arm1 failures list names $planted"
    else
        bad "arm1 raw log does not list $planted under failures:"
    fi
done
for target in '-p memra-tier --test contracts' '-p memra-kv --lib' '-p memra-cli --lib'; do
    if grep -qF -- "$target" "$log" "$raw"; then
        ok "arm1 cargo names the failed target $target (every crate ran)"
    else
        bad "arm1 failed target $target not named; did --no-fail-fast hold?"
    fi
done
if grep -q 'portable-suites: PASS' "$log"; then
    bad "arm1 wrapper printed PASS with planted failures"
else
    ok "arm1 no PASS line"
fi
rm -rf "$copy"

# ---- arm 2: an undeclared skipping test reds the static census before cargo -----------------
skip_arm() {
    # skip_arm <label> <expected libtest path>: the copy already carries the planted test.
    local label=$1 expected=$2 log=$tmp/$1.log rc
    run_wrapper "$log"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        ok "$label undeclared SKIP reds the wrapper (exit $rc)"
    else
        bad "$label wrapper exited 0 with an undeclared skipping test"
    fi
    if grep -q 'skip-census: FAIL' "$log" && grep -qF "memra-cli $expected (" "$log"; then
        ok "$label the static census names the planted test ($expected)"
    else
        bad "$label refusal does not come from the static census naming $expected"
    fi
    if grep -q 'skip-census: running:' "$log"; then
        bad "$label cargo ran although the static census should refuse first"
    else
        ok "$label refused before cargo ran"
    fi
}
# 2a: a new integration-test binary. libtest prints its tests with no prefix.
stage planted-skip-tests
mkdir -p "$copy/crates/memra-cli/tests"
cat > "$copy/crates/memra-cli/tests/planted_skip.rs" <<'EOF'
#[test]
fn planted_artifact_gated_integration_test() {
    eprintln!("SKIP: planted artifact absent (tools/test_portable_suites.sh, tests/)");
}
EOF
skip_arm arm2a planted_artifact_gated_integration_test
rm -rf "$copy"
# 2b: inside src, the shape the census was written for.
stage planted-skip-src
cat >> "$copy/crates/memra-cli/src/lib.rs" <<'EOF'

#[cfg(test)]
mod planted_skip {
    #[test]
    fn planted_artifact_gated_test() {
        eprintln!("SKIP: planted artifact absent (tools/test_portable_suites.sh, src/)");
    }
}
EOF
skip_arm arm2b planted_skip::planted_artifact_gated_test
rm -rf "$copy"

# ---- arm 3: wiring --------------------------------------------------------------------------
ci=$here/.github/workflows/ci.yml
if grep -q 'run: tools/portable-suites.sh' "$ci"; then
    ok "arm3 ci.yml runs tools/portable-suites.sh"
else
    bad "arm3 ci.yml does not run tools/portable-suites.sh"
fi
if grep -q 'run: tools/test_portable_suites.sh' "$ci"; then
    ok "arm3 ci.yml runs this fixture"
else
    bad "arm3 ci.yml does not run tools/test_portable_suites.sh"
fi
if grep -q 'tools/portable-suites.sh' "$here/tools/local-ci.sh"; then
    ok "arm3 tools/local-ci.sh runs tools/portable-suites.sh"
else
    bad "arm3 tools/local-ci.sh does not run tools/portable-suites.sh"
fi
# One executor (day 14). Comment lines stripped so prose cannot satisfy or trip these.
ci_live=$(grep -vE '^\s*#' "$ci")
local_live=$(grep -vE '^\s*#' "$here/tools/local-ci.sh")
jobs=$(printf '%s\n' "$ci_live" | grep -c '^  portable-suites:' || true)
if [[ $jobs -eq 1 ]]; then
    ok "arm3 ci.yml has exactly one portable-suites job"
else
    bad "arm3 ci.yml has $jobs portable-suites jobs (a duplicate key makes GitHub refuse the whole file)"
fi
if printf '%s\n%s\n' "$ci_live" "$local_live" | grep -E 'cargo test' | grep -qE 'memra-(tier|kv|cli)'; then
    bad "arm3 a live line of ci.yml or tools/local-ci.sh runs cargo test on the three crates outside the wrapper"
else
    ok "arm3 no live cargo test on memra-tier/-kv/-cli outside the wrapper (one executor)"
fi
forward=$here/tools/ci-portable.sh
if [[ -f "$forward" ]]; then
    if grep -vE '^\s*#' "$forward" | grep -q 'portable-suites.sh' && ! grep -vE '^\s*#' "$forward" | grep -q 'cargo'; then
        ok "arm3 tools/ci-portable.sh forwards to the wrapper and runs no cargo of its own"
    else
        bad "arm3 tools/ci-portable.sh is a second executor (must forward to portable-suites.sh and run no cargo)"
    fi
    if printf '%s\n%s\n' "$ci_live" "$local_live" | grep -q 'ci-portable.sh'; then
        bad "arm3 ci.yml or tools/local-ci.sh calls tools/ci-portable.sh instead of the wrapper"
    else
        ok "arm3 neither ci.yml nor tools/local-ci.sh calls tools/ci-portable.sh"
    fi
else
    ok "arm3 tools/ci-portable.sh is absent (no second name)"
    ok "arm3 neither ci.yml nor tools/local-ci.sh calls tools/ci-portable.sh"
fi

echo "test_portable_suites: $pass ok, $fail FAIL"
[[ $fail -eq 0 ]]
