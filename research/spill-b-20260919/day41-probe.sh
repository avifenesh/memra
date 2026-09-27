#!/usr/bin/env bash
# DAY41 1.2: concat-prime-probe primepath --hist K --rewind over docs/SERVING.md at each length, K in {32, 256}, a
# 64-token suffix, one run per cell under the rig lock (bounded wait). Never a signal to anything this lane did not start.
# usage: day41-probe.sh <out dir> <probe binary> <model>    env: RIG_LOCK (/tmp/memra-5090.lock), LENGTHS (6144,30720)
set -uo pipefail
OUT=${1:?out}; PROBE=${2:?probe}; MODEL=${3:?model}
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
LENGTHS=${LENGTHS:-6144,30720}
mkdir -p "$OUT"
SRC=$WT/docs/SERVING.md
# The suffix: a later stretch of the same document (the next turn's bytes), cut to 64 tokens by the probe.
tail -c 4000 "$SRC" > "$OUT/suffix.txt"
sha256sum "$SRC" "$OUT/suffix.txt" "$PROBE" > "$OUT/inputs.sha256"
IFS=, read -r -a lens <<< "$LENGTHS"
for L in "${lens[@]}"; do
  for K in 32 256; do
    cell=$OUT/primepath-L$L-K$K.log
    [ -s "$cell" ] && command grep -q "^verdict rewind" "$cell" && continue
    nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm --format=csv > "$OUT/gpu-before-L$L-K$K.csv" 2>&1
    flock -w 7200 "$RIG_LOCK" "$PROBE" "$MODEL" primepath --prompt-a "@$SRC" --prompt-tokens "$L" --suffix "@$OUT/suffix.txt" \
      --suffix-tokens 64 --hist "$K" --rewind --steps 48 > "$cell" 2>&1
    echo "exit=$?" >> "$cell"
    sleep "${YIELD_S:-0}" # the lane yields the card between cells when YIELD_S is set (lead, 2026-09-26)
    nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm --format=csv > "$OUT/gpu-after-L$L-K$K.csv" 2>&1
  done
done
command grep -h "^verdict\|^cost\|^primepath:" "$OUT"/primepath-*.log > "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"
