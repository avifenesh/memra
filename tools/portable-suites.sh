#!/usr/bin/env bash
# portable-suites.sh: EXECUTE the tier, KV and onboarding-CLI test suites (memra #545).
#
# WHY. Until 2026-09-21 no standing gate ran these suites. ci.yml compiled them (build, clippy)
# and executed memra-gguf/-reference/-tokenizer/-validate/-sampling, memra-lanes, memra-server
# and memra-engine; tools/local-ci.sh executed server, engine and gguf. The 325 tests across
# memra-tier (contracts, storage, bank, peer, placement, reclaim, four compile-fail doctests),
# memra-kv (hierarchy, materializers, governor, contracts_v12) and memra-cli (onboarding
# receipts) ran only when a lane ran them by hand: retirement, cancellation and ownership rules
# with red arms that nothing in CI ever executed.
#
# WHAT. One command, the same text in .github/workflows/ci.yml (job portable-suites),
# tools/local-ci.sh (the CPU chain) and tools/test_portable_suites.sh (the teeth: a planted
# failing tier/KV/CLI test in a copy of the tree must red THIS script, so build and clippy alone
# can never satisfy the gate). It runs through tools/skip-census.py, so a test that skips for a
# missing artifact is printed and counted against a budget of ZERO; today none of the three
# crates has such a test (the static census below is the proof, in both directions).
#   * no --lib: memra-tier's six integration suites and its doctests are most of the tests
#   * --no-fail-fast: every test binary runs, so a red names every failing suite, not the first
#   * --offline: the resolution is the committed Cargo.lock; the caller warms the registry
#     (`cargo fetch --locked` in ci.yml; the rig's cache locally)
#   * the job count is cargo's own CARGO_BUILD_JOBS / .cargo/config.toml; no knob here
#   * --min-passed 300: the non-vacuity floor (325 measured on 2026-09-21); it moves with the suite
#
# CPU only, GPU-less by construction: memra-tier has no CUDA dependency, memra-kv links cudarc
# with dynamic-loading and no test opens the driver, memra-cli depends on gguf/reference/
# tokenizer. A green here is EXECUTION of CPU suites, not hardware qualification: it promotes no
# model, numeric program, default or support state (CLAUDE.md, the three support states).
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")/.." && pwd)
cd "$here"

crates=(memra-tier memra-kv memra-cli)
verify_args=()
for crate in "${crates[@]}"; do
    verify_args+=(--crate "$crate")
done

echo "portable-suites: static skip census over ${crates[*]} (an artifact-gated #[test] must be declared)"
python3 tools/skip-census.py verify "${verify_args[@]}"

echo "portable-suites: cargo test -p memra-tier -p memra-kv -p memra-cli --offline --no-fail-fast"
echo "portable-suites: CPU execution of the tier, KV and onboarding CLI suites; NOT GPU qualification"
# The raw cargo output is banked (evidence discipline: never summary-only), under target/ so it
# is never tracked; tools/test_portable_suites.sh reads it from its planted copy.
mkdir -p "$here/target"
MEMRA_PORTABLE_SKIP_BUDGET=0 python3 tools/skip-census.py run \
    --budget-var MEMRA_PORTABLE_SKIP_BUDGET --min-passed 300 \
    --log "$here/target/portable-suites.log" \
    -- cargo test -p memra-tier -p memra-kv -p memra-cli --offline --no-fail-fast

echo "portable-suites: PASS (tier, KV and onboarding CLI suites executed on CPU; skips 0 of budget 0; NOT GPU qualification)"
