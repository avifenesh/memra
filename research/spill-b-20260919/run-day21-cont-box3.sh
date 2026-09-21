#!/usr/bin/env bash
# Day 21 memra#427 reproduction cell, target card (run from /root/wt-b on the box): qwen-a4-continuation-gate on
# the lane tip binary, through the collector with the inherited canonical lock (--rig pro-single,
# /tmp/memra-gpu.lock). Totals 9296 (tails 16 48 80), 9297 (17), 9312 (32): the differing split, its ok
# neighbours and the bracket at the same aligned head 9280. MEMRA_PRIME_ROW_RECEIPT=1 is the existing per-call
# digest diagnostic. Pass/fail cell, not timed. Bounded lock retries, never kills a holder.
# usage: run-day21-cont-box3.sh <cell-name> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; tmo=${2:?timeout}
R=${R:-/root/spill-receipts/b-day21}
G=${G:-$R/bins/tip/qwen-a4-continuation-gate}
M=${M:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
P=${P:-/root/wt-b/docs/SERVING.md}
cd /root/wt-b || exit 1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-before.csv"
body='set -uo pipefail; G="$1"; M="$2"; P="$3"; O="$4"; mkdir -p "$O"; rc=0
for spec in "9296 16 48 80" "9297 17" "9312 32"; do
  set -- $spec; total=$1; shift
  echo "== total=$total tails=$*"
  MEMRA_PRIME_ROW_RECEIPT=1 "$G" "$M" "$P" "$total" "$@" > "$O/total-$total.log" 2>&1; r=$?
  echo "exit=$r" >> "$O/total-$total.log"; grep -v "^\[prime-row\]" "$O/total-$total.log"; [ $r -ne 0 ] && rc=1
done
exit $rc'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  python3 tools/tier-battery.py --rig pro-single --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" cont "$G" "$M" "$P" "$out/cell" > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; sha256sum "$G" "$M" > "$out/binary.sha256"; sha256sum "$P" > "$out/prompt.sha256"; }
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$R/$cell-compute-apps-after.csv"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
