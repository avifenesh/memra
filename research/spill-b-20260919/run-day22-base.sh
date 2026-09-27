#!/usr/bin/env bash
# Day 22 memra#427 cell F5 (the (f) check of the resumed brief), local RTX 5090: the BASE-tree continuation gate
# (the fix's three source hunks reverted on the merged tree, nothing else; build receipt `build-base/`) on the same
# artifact and prompt as `fix-gate/`, three arms: the default schedule at 9296 (no 16-row chunk in the one-call
# prime), `MEMRA_PRIME_CHUNK=32` at 9296 (the one-call prime ends in a 16-row chunk), and 9297 (17-row tail).
# Read against `fix-gate/`: the arm whose schedule ends in a 16-row chunk is expected to differ between base and fix,
# with the fix value equal to the wide-chunk digest; the other two are expected identical. Pass/fail digests, not
# timed. Bounded lock retries, never kills a holder.
# usage: run-day22-base.sh <cell-name> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}
WT=${WT:-$HOME/projects/wt-spill-b}
R=${R:-$WT/research/spill-b-20260919/rtx5090-day22}
GATE=${GATE:-$WT/target/bins/day22-base/qwen-a4-continuation-gate}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
PROMPT=${PROMPT:-$WT/docs/SERVING.md}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
mkdir -p "$R"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; G="$1"; M="$2"; P="$3"; O="$4"; mkdir -p "$O"
arm() { name="$1"; shift; total="$1"; shift; tails="$1"; shift
  echo "== arm $name total=$total tails=$tails env: $*"
  env MEMRA_PRIME_ROW_RECEIPT=1 "$@" "$G" "$M" "$P" "$total" $tails > "$O/arm-$name.log" 2>&1
  echo "exit=$?" >> "$O/arm-$name.log"; grep -v "^\[prime-row\]" "$O/arm-$name.log"; }
arm B-9296 9296 "16 48"
arm B-9296-chunk32 9296 "16 48" MEMRA_PRIME_CHUNK=32
arm B-9297 9297 "17"
exit 0'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" basegate "$GATE" "$MODEL" "$PROMPT" "$out/cell" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$GATE" "$MODEL" > "$out/binary.sha256"; sha256sum "$PROMPT" > "$out/prompt.sha256"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
