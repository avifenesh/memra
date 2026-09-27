#!/usr/bin/env bash
# Day 23 copy of the day-22/21 runner for the MERGED tree (origin/main 653c997f4, #614 small_m_tier_max; the lane's prefill_rows scope removed): same cell, same arms, receipts under rtx5090-day23/.
# Day 22 memra#427 cell W, local RTX 5090: the per-op width walk. `qwen-a4-width-walk` calls Engine::matmul_prefill
# on every projection the prime walk multiplies, at the reference width 17 and at 16 and 48, over one deterministic
# activation, and compares the shared leading rows bitwise. Predictions are in DAY22.md section 1.2, written before
# this ran. No flag set. Pass/fail digests, not timed. Bounded lock retries, never kills a holder.
# usage: run-day22-walk.sh <cell-name> <collector-timeout-s> [ref] [widths...]
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}; shift 2
WT=${WT:-$HOME/projects/wt-spill-b}
R=${R:-$WT/research/spill-b-20260919/rtx5090-day23}
WALK=${WALK:-$WT/target/release/qwen-a4-width-walk}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
mkdir -p "$R"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; B="$1"; M="$2"; O="$3"; shift 3; mkdir -p "$O"
"$B" "$M" "$@" > "$O/walk.log" 2>&1; echo "exit=$?" >> "$O/walk.log"
grep -c "" "$O/walk.log"; grep "DIFFERS\|^WIDTH WALK\|activation program\|exit=" "$O/walk.log"'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" walk "$WALK" "$MODEL" "$out/cell" "$@" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$WALK" "$MODEL" > "$out/binary.sha256"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
