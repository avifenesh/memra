#!/usr/bin/env bash
# Day 21 memra#427 reproduction cell, local RTX 5090: qwen-a4-continuation-gate (the reproducer named in the
# issue, on main since #424) on the lane tip binary, through the collector with the inherited canonical lock
# (`--rig rtx5090`, `/tmp/memra-5090.lock`). Four totals so the 16-row tail is bracketed at the same aligned
# head 9280: 9296 (tails 16, 48, ..., 208: the issue's table), 9297 (tails 17, 49), 9311 (tails 31, 63),
# 9312 (tails 32, 64, 96: whole-WY-chunk tails). MEMRA_PRIME_ROW_RECEIPT=1 (existing diagnostic) prints a
# `[prime-row]` digest per prime call. Pass/fail cell, not timed. Bounded lock retries, never kills a holder.
# usage: run-day21-cont.sh <cell-name> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}
WT=${WT:-$HOME/projects/wt-spill-b}
R=${R:-$WT/research/spill-b-20260919/rtx5090-day21}
GATE=${GATE:-$WT/target/release/qwen-a4-continuation-gate}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
PROMPT=${PROMPT:-$WT/docs/SERVING.md}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; G="$1"; M="$2"; P="$3"; O="$4"; mkdir -p "$O"; rc=0
for spec in "9296 16 48 80 112 144 176 208" "9297 17 49" "9311 31 63" "9312 32 64 96"; do
  set -- $spec; total=$1; shift
  echo "== total=$total tails=$*"
  MEMRA_PRIME_ROW_RECEIPT=1 "$G" "$M" "$P" "$total" "$@" > "$O/total-$total.log" 2>&1; r=$?
  echo "exit=$r" >> "$O/total-$total.log"; grep -v "^\[prime-row\]" "$O/total-$total.log"; [ $r -ne 0 ] && rc=1
done
exit $rc'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" cont "$GATE" "$MODEL" "$PROMPT" "$out/cell" > "$out-driver.log" 2>&1
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
