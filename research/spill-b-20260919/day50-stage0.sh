#!/usr/bin/env bash
# DAY50 stage 0 (DAY50.md 1.2): per L, one timed `callcost` run (the wall readings), then the same run under
# `nsys profile` (the kernel trace), then day50-trace.py. Each run holds the rig lock (bounded wait) and the lock stays
# free for YIELD_S after it. Never a signal to anything this lane did not start.
# usage: day50-stage0.sh <out dir> <probe binary> <model> <card label>
# env: RIG_LOCK (/tmp/memra-5090.lock), LENGTHS (6144,30720), ROWS (32,64,288), REPS (5), YIELD_S (240).
set -uo pipefail
OUT=${1:?out}; PROBE=${2:?probe}; MODEL=${3:?model}; CARD=${4:?card}
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
LENGTHS=${LENGTHS:-6144,30720}
ROWS=${ROWS:-32,64,288}
REPS=${REPS:-5}
YIELD_S=${YIELD_S:-240}
mkdir -p "$OUT"
SRC=$WT/docs/SERVING.md
sha256sum "$SRC" "$PROBE" "$MODEL" > "$OUT/inputs.sha256" 2>/dev/null || sha256sum "$SRC" "$PROBE" > "$OUT/inputs.sha256"
log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
IFS=, read -r -a lens <<< "$LENGTHS"
for L in "${lens[@]}"; do
  args=("$MODEL" callcost --prompt-a "@$SRC" --prompt-tokens "$L" --rows "$ROWS" --reps "$REPS" --gap-ms 50)
  if ! command grep -q "^callcost L=$L R=" "$OUT/wall-L$L.log" 2>/dev/null; then
    nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm --format=csv > "$OUT/gpu-before-L$L.csv" 2>&1
    flock -w 7200 "$RIG_LOCK" "$PROBE" "${args[@]}" > "$OUT/wall-L$L.log" 2>&1
    rc=$?
    echo "exit=$rc" >> "$OUT/wall-L$L.log"
    log "wall L=$L rc=$rc"
    sleep "$YIELD_S"
  fi
  if [ ! -s "$OUT/trace-L$L.csv" ]; then
    flock -w 7200 "$RIG_LOCK" nsys profile -t cuda -o "$OUT/nsys-L$L" --force-overwrite true "$PROBE" "${args[@]}" \
      > "$OUT/nsys-L$L.log" 2>&1
    rc=$?
    echo "exit=$rc" >> "$OUT/nsys-L$L.log"
    log "nsys L=$L rc=$rc"
    nsys stats --report cuda_gpu_trace --format csv --output "$OUT/trace-L$L" "$OUT/nsys-L$L.nsys-rep" > "$OUT/stats-L$L.log" 2>&1
    rc=$?
    log "stats L=$L rc=$rc"
    # nsys names the csv <output>_cuda_gpu_trace.csv
    [ -s "$OUT/trace-L${L}_cuda_gpu_trace.csv" ] && mv "$OUT/trace-L${L}_cuda_gpu_trace.csv" "$OUT/trace-L$L.csv"
    sleep "$YIELD_S"
  fi
  python3 "$WT/research/spill-b-20260919/day50-trace.py" "$CARD" "$L" "$ROWS" "$REPS" "$OUT/trace-L$L.csv" \
    "$OUT/nsys-L$L.log" > "$OUT/read-L$L.log" 2>&1
  command grep -h "^callcost L=" "$OUT/wall-L$L.log" >> "$OUT/read-L$L.log"
done
cat "$OUT"/read-L*.log > "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"
