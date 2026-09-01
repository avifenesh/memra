#!/bin/bash
# The H100 serving-lane validation battery (ARCHITECTURE-H100.md §6) — one command,
# gates-first, perf after. Run on the box from the repo root. Exits nonzero on any gate
# failure; perf numbers are recorded, not judged (bands live in the docs).
#
# Usage: tools/validate-h100.sh <model.gguf> [--quick]
#   --quick: 16-step gates, skip the bench curve (pre-commit sanity).
#
# GATE-INTEGRITY (2026-08-19, round 2 — GATE-INTEGRITY-20260819.md A-1/A-2/A-5):
# `set -o pipefail` is load-bearing here, not hygiene. Every gate below is shaped
# `<binary> | tail -1 | grep -q "ALL GREEN"`, and WITHOUT pipefail the pipeline's status is
# `grep`'s alone: a binary that dies on its second cell still hands `tail` whatever it printed
# first, and a `grep` that matches an early "ALL GREEN" from a sub-cell returns 0. With
# pipefail a nonzero producer fails the pipeline no matter what the tail matched.
set -u
set -o pipefail
MODEL="${1:?model.gguf}"
QUICK=""
[ "${2:-}" = "--quick" ] && QUICK=1
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
NVCC="${MEMRA_NVCC:-/usr/local/cuda-13.1/bin/nvcc}"
STEPS=$([ -n "$QUICK" ] && echo 16 || echo 32)
FAIL=0

# kernel-check's oracle sections (dtype5 / NVFP4 / Q8MMQ / G12 / G27) resolve their models
# through MEMRA_KC_MODELS_DIR and SKIP silently when it is unset off the legacy path — the
# receipted "rounds 44-47 ran the battery blind on exactly the models the lane fights over"
# incident. Honor the battery's own var so this box searches where local-ci.sh searches; the
# skip budget below is what actually enforces the outcome.
if [ -z "${MEMRA_KC_MODELS_DIR:-}" ] && [ -n "${MEMRA_MODELS_DIR:-}" ]; then
  export MEMRA_KC_MODELS_DIR="$MEMRA_MODELS_DIR"
fi
# Skipped kernel-check cells are FATAL by default. `ALL GREEN (N cells, M skipped)` reads as
# green to a `grep -q "ALL GREEN"`, so a box missing every oracle model printed the same
# banner as a box that ran them all. Raising the budget is allowed, but it has to NAME the
# number, and the run says so in its output — an accounted skip, not an invisible one.
KC_SKIP_BUDGET="${MEMRA_H100_KC_SKIP_BUDGET:-0}"

echo "== build (sm_90a; cu sources touched to defeat rsync-stale fatbins) =="
touch crates/memra-engine/cu/*.cu crates/memra-engine/build.rs
MEMRA_CUDA_ARCH=90a MEMRA_NVCC=$NVCC cargo build --release -p memra-engine \
  --bin kernel-check --bin run-gen --bin decode-batch-gate --bin decode-batch-bench \
  || exit 1
MEMRA_CUDA_ARCH=90a MEMRA_NVCC=$NVCC cargo build --release -p memra-server || exit 1

# ---------------------------------------------------------------------------
# UNIT SUITES — a real gate, not a printed tail.
#
# This was `cargo test ... | tail -1`: no pipefail, no verdict grep, FAIL never touched. The
# entire engine unit suite could be red and line "ALL GATES GREEN" still printed with exit 0 —
# a gate that reported success while DISCARDING the evidence it had just produced.
#
# Three assertions per suite, because exit 0 alone is not enough:
#   1. cargo's own exit status is 0;
#   2. every `test result:` line says `ok.` (a FAILED suite among several is caught);
#   3. the run was not VACUOUS — total passed > 0 and `filtered out` is 0 on every line.
#      (3) is the ci.yml shape: `cargo test -p memra-engine cpu_experts --lib` is a NAME
#      FILTER, and a filter that stops matching leaves a green "0 passed; N filtered out".
#
# Which crates: memra-engine (--lib, the policy/kernel-policy tests) plus the two suites that
# had NO caller in any gate or workflow at all — memra-server (13 files, the request-contract
# and ledger invariants) and memra-tokenizer (5 files, incl. the pre-tokenizer resolution that
# decides whether a model loads). Both are CPU-only; this box has the GPU but does not need it
# here. See .github/workflows/ci.yml for the four dependency-free crates' suites.
# ---------------------------------------------------------------------------
# Logs are the evidence, so they are NOT deleted on the failure path (that is the whole point
# of this change); they ARE deleted on the green path, per the /tmp hygiene law.
UNIT_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/validate-h100-gate-XXXXXX")
trap 'rm -rf "$UNIT_LOG_DIR"' INT TERM

unit_suite() { # $1 crate  [$2.. cargo target selector, e.g. --lib]
  local crate=$1 log slug rc results bad_results passed filtered
  shift
  # The selector is part of the log name: this function is called twice for memra-engine (--lib
  # and the parity bins) and one log silently overwriting the other would discard the evidence
  # this whole change exists to keep.
  slug=$(printf '%s %s' "$crate" "$*" | tr -c 'A-Za-z0-9._-' '-')
  log="$UNIT_LOG_DIR/$slug.log"
  echo "== gate: unit suite ($crate $*) =="
  MEMRA_CUDA_ARCH=90a MEMRA_NVCC=$NVCC cargo test --release -p "$crate" "$@" > "$log" 2>&1
  rc=$?
  results=$(grep -c '^test result:' "$log")
  bad_results=$(grep -c '^test result: FAILED' "$log")
  passed=$(awk '/^test result:/ { for (i = 1; i <= NF; i++) if ($(i+1) ~ /^passed/) s += $i } END { print s + 0 }' "$log")
  filtered=$(awk '/^test result:/ { for (i = 1; i <= NF; i++) if ($(i+1) == "filtered") s += $i } END { print s + 0 }' "$log")
  if [ "$rc" -ne 0 ] || [ "$bad_results" -ne 0 ]; then
    echo "UNIT-SUITE($crate) FAIL (cargo rc=$rc, $bad_results failed result line(s))"
    grep -E '^(test .* FAILED|failures:|error(\[|:))' "$log" | head -40
    tail -20 "$log"
    FAIL=1
    return 1
  fi
  # FILTERED before VACUOUS, deliberately: a filtered-to-nothing run is ALSO vacuous, and the
  # more specific diagnosis is the useful one — "a name filter matched nothing" tells you where
  # to look, "0 tests ran" does not.
  if [ "$filtered" -ne 0 ]; then
    echo "UNIT-SUITE($crate) FAIL — $filtered test(s) FILTERED OUT of an unfiltered run."
    echo "  This gate must run the whole suite; a name filter is how a suite rots green:"
    echo "  the day the filtered name moves, the run prints '0 passed; N filtered out' and exits 0."
    grep '^test result:' "$log" | sed 's/^/    /'
    FAIL=1
    return 1
  fi
  if [ "$results" -eq 0 ] || [ "$passed" -eq 0 ]; then
    echo "UNIT-SUITE($crate) FAIL — VACUOUS: $results result line(s), $passed passed."
    echo "  A suite that ran no tests is not a green suite. Last 20 lines:"
    tail -20 "$log"
    FAIL=1
    return 1
  fi
  echo "unit suite($crate): $passed passed, 0 failed, 0 filtered out ($results result line(s))"
}

# memra-server takes no `--lib`: it is a binary-only crate and `--lib` exits 101 with
# "no library targets found in package". Its tests live in the bin targets.
unit_suite memra-engine --lib
unit_suite memra-server
unit_suite memra-tokenizer --lib

# ---------------------------------------------------------------------------
# memra-gguf THROUGH THE SKIP CENSUS (GATE-INTEGRITY-20260819 section 10).
#
# This crate is not run by unit_suite, because unit_suite cannot see the hole it has: twelve
# #[test] fns guard on an artifact and `eprintln!("SKIP..."); return;`, and the test PASSES.
# `90 passed` is printed whether or not one model-backed assertion ran — including
# nv27b_twin_parity, where the n_rot rotary-width geometry check lands. The suite's verdict is
# perfectly green and says nothing.
#
# The census asserts the suite's own verdict FIRST (exit status, every `test result: ok.`, no
# filtering, not vacuous — the unit_suite assertions), then counts the SKIPs against a named
# budget. Budget 0 by default, deliberately: this box HAS the models, so a skip here means an
# artifact went missing and the battery should say so. It CAN newly red a box that was quietly
# skipping, which is the point, and the refusal names the number to set.
#
# Reference measurement on the dev rig 2026-08-20: 90 passed, 7 skipped (minimax-m3-nvfp4-reap50
# and hy3-reap50-q4k-memra not staged, /tmp/iq3s_raw.bin absent).
# ---------------------------------------------------------------------------
GGUF_SKIP_BUDGET="${MEMRA_H100_GGUF_SKIP_BUDGET:-0}"
echo "== gate: unit suite (memra-gguf --lib) + skip census (budget $GGUF_SKIP_BUDGET) =="
if ! python3 tools/skip-census.py verify > "$UNIT_LOG_DIR/skip-census-verify.log" 2>&1; then
  echo "SKIP-CENSUS VERIFY FAIL — tools/skip-census.tsv disagrees with the source."
  cat "$UNIT_LOG_DIR/skip-census-verify.log"
  FAIL=1
elif ! MEMRA_GGUF_SKIP_BUDGET="$GGUF_SKIP_BUDGET" python3 tools/skip-census.py run \
      --budget-var MEMRA_GGUF_SKIP_BUDGET --min-passed 80 \
      --log "$UNIT_LOG_DIR/memra-gguf-census.log" \
      -- cargo test --release -p memra-gguf --lib; then
  echo "  (raise it deliberately with MEMRA_H100_GGUF_SKIP_BUDGET=<n> if the artifacts are"
  echo "   genuinely not staged on this box)"
  FAIL=1
fi
# The parity-geometry rule lives in a module included by the two parity binaries, so `--lib`
# cannot reach it; the targets have to be named. `--bin X` is a TARGET selector, not a name
# filter — every test in those targets runs, and `filtered out` stays 0.
unit_suite memra-engine --bin dflash_parity --bin dspark_q38_parity

# ---------------------------------------------------------------------------
# kernel-check — manifest-pinned, and skips are accounted.
#
# Was `./target/release/kernel-check | tail -1 | grep -q "ALL GREEN"`. Two holes:
#   * no --require-manifest, which local-ci.sh passes: a run that never REACHED the 27B or
#     step35 cells still ends `ALL GREEN`, because the banner counts what ran.
#   * `ALL GREEN (N cells, M skipped)` matches `grep -q "ALL GREEN"` for any M.
# Now: both manifests required, the last line matched EXACTLY against the verdict shape, the
# SKIP lines printed, and M compared against a named budget.
# ---------------------------------------------------------------------------
echo "== gate: kernel-check (manifest-pinned, skip budget $KC_SKIP_BUDGET) =="
KC_LOG="$UNIT_LOG_DIR/kernel-check.log"
if ! ./target/release/kernel-check \
      --require-manifest tools/kernel-check-27b.cells \
      --require-manifest tools/kernel-check-step35.cells > "$KC_LOG" 2>&1; then
  echo "KERNEL-CHECK FAIL (nonzero exit)"
  grep -E 'MISSING REQUIRED CELL|FAIL' "$KC_LOG" | head -20
  tail -10 "$KC_LOG"
  FAIL=1
else
  KC_LAST=$(tail -1 "$KC_LOG")
  if ! printf '%s\n' "$KC_LAST" \
       | grep -qE '^ALL GREEN \([0-9]+ cells, [0-9]+ skipped\)$'; then
    echo "KERNEL-CHECK FAIL — no verdict line; last line was: ${KC_LAST:-<empty>}"
    tail -10 "$KC_LOG"
    FAIL=1
  else
    KC_CELLS=$(printf '%s\n' "$KC_LAST" | sed -E 's/^ALL GREEN \(([0-9]+) cells.*/\1/')
    KC_SKIPPED=$(printf '%s\n' "$KC_LAST" | sed -E 's/.*, ([0-9]+) skipped\)$/\1/')
    grep '^SKIP ' "$KC_LOG" | sed 's/^/  /' || true
    if [ "$KC_SKIPPED" -gt "$KC_SKIP_BUDGET" ]; then
      echo "KERNEL-CHECK FAIL — $KC_SKIPPED cell(s) skipped, budget $KC_SKIP_BUDGET."
      echo "  Skipped cells are the receipted H100 rounds-44-47 shape: the battery reported"
      echo "  green on exactly the models it never loaded. Stage the models (see"
      echo "  MEMRA_KC_MODELS_DIR=${MEMRA_KC_MODELS_DIR:-<unset>}) or set"
      echo "  MEMRA_H100_KC_SKIP_BUDGET=$KC_SKIPPED to account for them deliberately."
      FAIL=1
    else
      echo "kernel-check: $KC_CELLS cells, $KC_SKIPPED skipped (budget $KC_SKIP_BUDGET)"
    fi
  fi
fi

echo "== gate: decode-batch (config B=8) =="
./target/release/decode-batch-gate "$MODEL" --steps $STEPS --batch 8 --mode config \
  | tail -1 | grep -q "ALL GREEN" || { echo "BATCH-GATE(config) FAIL"; FAIL=1; }

echo "== gate: decode-batch (strict, equalized composition) =="
MEMRA_SERVE_B1FAST=1 MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1 \
  ./target/release/decode-batch-gate "$MODEL" \
  --steps $STEPS --batch 4 --mode strict \
  | tail -1 | grep -q "ALL GREEN" || { echo "BATCH-GATE(strict) FAIL"; FAIL=1; }

# GRAPH LANE gates (round 35: graph-decode-gate rotted OUTSIDE this battery for weeks —
# an emission off-by-one in the gate masqueraded as 171/256 stream corruption. Everything
# guarding a live lane belongs HERE.)
echo "== gate: decode-dc (device counters, bit-identity) =="
# A silenced build with no status check runs the PREVIOUS binaries — the stale-artifact rot
# this file's own comment calls "the rot mode the H100 lane's law 3 warns about".
if ! MEMRA_CUDA_ARCH=90a MEMRA_NVCC=$NVCC cargo build --release -p memra-engine \
     --bin decode-dc-gate --bin graph-decode-gate --bin graph-session-gate \
     > "$UNIT_LOG_DIR/graph-bins-build.log" 2>&1; then
  echo "GRAPH-BINS BUILD FAIL — refusing to gate on stale binaries"
  tail -20 "$UNIT_LOG_DIR/graph-bins-build.log"
  FAIL=1
fi
./target/release/decode-dc-gate "$MODEL" 2>&1 | tail -1 | grep -q "PASS" \
  || { echo "DC-GATE FAIL"; FAIL=1; }
echo "== gate: graph-decode (capture/replay bit-identity) =="
./target/release/graph-decode-gate "$MODEL" 2>&1 | tail -1 | grep -q "PASS" \
  || { echo "GRAPH-DECODE-GATE FAIL"; FAIL=1; }
echo "== gate: graph-session (serving GraphSession vs generate_graph) =="
./target/release/graph-session-gate "$MODEL" 2>&1 | tail -1 | grep -q "ALL GREEN" \
  || { echo "GRAPH-SESSION-GATE FAIL"; FAIL=1; }

if [ -z "$QUICK" ] && [ $FAIL -eq 0 ]; then
  echo "== perf record: serving-regime curve (ctx=512, N=3) =="
  ./target/release/decode-batch-bench "$MODEL" --steps 96 --reps 3 --batches 1,2,4,8 --ctx 512 \
    | grep -E "B=|scale"
  echo "== perf record: single-seq prime+decode (N=3) =="
  # tee the raw log; never let the pipe swallow error output (evidence discipline)
  bash tools/bench_memra_protocol.sh "$MODEL" 3 512 2>&1 | tee memra-single.log \
    | grep -E "run [0-9]|median" || echo "single-seq bench produced no readings — see memra-single.log"
fi

if [ $FAIL -eq 0 ]; then
  rm -rf "$UNIT_LOG_DIR"
  echo "VALIDATE-H100: ALL GATES GREEN"
else
  echo "VALIDATE-H100: FAILURES ($FAIL) — gate logs kept at $UNIT_LOG_DIR"
fi
exit $FAIL
