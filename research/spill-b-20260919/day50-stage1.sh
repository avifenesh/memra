#!/usr/bin/env bash
# DAY50 stage 1 (DAY50.md addendum D): `kvrow` (the KV bytes per row and the D2D copy of rows [0, g)) and `overlap` (engine
# A's decode TPOT alone, beside engine B's settle-shaped prime calls on a second Engine, and alone again) from the stage-0
# probe binary. Each run holds the rig lock first, checks the card idle under the hold (rig-hold.sh), runs with the
# hold's fd closed, releases, and yields YIELD_S. Never a signal to anything this lane did not start.
# usage: day50-stage1.sh <out dir> <probe binary> <model> <card label>
# env: WT, RIG_LOCK (/tmp/memra-5090.lock), CONTEXTS (6144,30720), LENGTHS (6144,30720), STEPS (64), REPS (5),
#      YIELD_S (240), MIN_HOST_GB (16).
set -uo pipefail
OUT=${1:?out}; PROBE=${2:?probe}; MODEL=${3:?model}; CARD=${4:?card}
WT=${WT:-$HOME/projects/wt-spill-b}
RIG_LOCK=${RIG_LOCK:-/tmp/memra-5090.lock}
CONTEXTS=${CONTEXTS:-6144,30720}
LENGTHS=${LENGTHS:-6144,30720}
STEPS=${STEPS:-64}
REPS=${REPS:-5}
YIELD_S=${YIELD_S:-240}
export MIN_HOST_GB=${MIN_HOST_GB:-16}
mkdir -p "$OUT"
cd "$WT" || exit 1
SRC=$WT/docs/SERVING.md
sha256sum "$SRC" "$PROBE" "$MODEL" > "$OUT/inputs.sha256"
log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
# shellcheck source=rig-hold.sh
. research/spill-b-20260919/rig-hold.sh
run_held() { # <name> <probe args...>
  local name=$1; shift
  if command grep -q "^exit=0" "$OUT/$name.log" 2>/dev/null; then return 0; fi
  rig_hold "$name" $((SECONDS + 4 * 3600))
  local r=$?
  [ $r = 0 ] || { log "$name: rig not held idle within 4 h; not run"; return 3; }
  nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$OUT/gpu-before-$name.csv" 2>&1
  "$PROBE" "$@" > "$OUT/$name.log" 2>&1 8>&-
  local rc=$?
  echo "exit=$rc" >> "$OUT/$name.log"
  rig_release "$name"
  log "$name rc=$rc"
  sleep "$YIELD_S"
}
run_held kvrow "$MODEL" kvrow --prompt-a "@$SRC" --contexts "$CONTEXTS" --reps "$REPS" --gap-ms 50
IFS=, read -r -a lens <<< "$LENGTHS"
for L in "${lens[@]}"; do
  run_held "overlap-L$L" "$MODEL" overlap --prompt-a "@$SRC" --prompt-tokens "$L" --rows 32 --decode-steps "$STEPS"
done
{ echo "card=$CARD"; command grep -h "^kvrow\|^overlap" "$OUT"/*.log; } > "$OUT/SUMMARY.txt"
cat "$OUT/SUMMARY.txt"
