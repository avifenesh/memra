#!/usr/bin/env bash
# Day 21 memra#427 seam bisect, local RTX 5090: the same gate binary and artifact as run-day21-cont.sh, total
# 9296 with tails 16 and 48 (the differing split and its nearest ok neighbour), once per existing rollback seam:
# baseline; MEMRA_GDN_CHUNKED=0 (sequential GDN scan, no WY chunking); MEMRA_F16OUT=0 (the t>=16 f16 epilogue
# off); MEMRA_NOFA=1 (naive attention kernel); MEMRA_MMQ_W4A8=0 (int8 GEMM class off). Every arm is read from
# its own log; an arm that turns the 16-row split `ok` names the kernel family the seam lives in. No new flag,
# no code change. Pass/fail cell, not timed. Bounded lock retries, never kills a holder.
# usage: run-day21-cont-bisect.sh <cell-name> <collector-timeout-s>
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
body='set -uo pipefail; G="$1"; M="$2"; P="$3"; O="$4"; mkdir -p "$O"
for arm in "baseline" "MEMRA_GDN_CHUNKED=0" "MEMRA_F16OUT=0" "MEMRA_NOFA=1" "MEMRA_MMQ_W4A8=0"; do
  echo "== arm $arm"
  if [ "$arm" = baseline ]; then env MEMRA_PRIME_ROW_RECEIPT=1 "$G" "$M" "$P" 9296 16 48 > "$O/arm-$arm.log" 2>&1
  else env MEMRA_PRIME_ROW_RECEIPT=1 "$arm" "$G" "$M" "$P" 9296 16 48 > "$O/arm-$arm.log" 2>&1; fi
  echo "exit=$?" >> "$O/arm-$arm.log"; grep -v "^\[prime-row\]" "$O/arm-$arm.log"
done'
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" \
    --execute bash -c "$body" bisect "$GATE" "$MODEL" "$PROMPT" "$out/cell" > "$out-driver.log" 2>&1
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
