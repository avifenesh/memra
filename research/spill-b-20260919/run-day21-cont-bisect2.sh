#!/usr/bin/env bash
# Day 21 memra#427 seam bisect, part 2, local RTX 5090: the chunk-width arms. MEMRA_PRIME_CHUNK=16 makes every call in
# both arms a 16-row chunk; =32 makes the one-call prime end in a 16-row chunk of its own (9296 = 290 x 32 + 16; a tail of
# exactly PRIME_MIN_T is not folded, the fold rule is strictly below); =48 leaves only the split head ending in a 16-row
# chunk (9280 = 193 x 48 + 16). If a 16-row chunk is its own program: 16 -> ok, 32 -> ok, 48 -> DIFFERS. Same gate,
# artifact and prompt as part 1; MEMRA_PRIME_CHUNK is a documented memory knob (FLAGS.md). Pass/fail cell, not timed.
# Bounded lock retries, never kills a holder.
# usage: run-day21-cont-bisect2.sh <cell-name> <collector-timeout-s>
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
for arm in "MEMRA_PRIME_CHUNK=16" "MEMRA_PRIME_CHUNK=32" "MEMRA_PRIME_CHUNK=48"; do
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
