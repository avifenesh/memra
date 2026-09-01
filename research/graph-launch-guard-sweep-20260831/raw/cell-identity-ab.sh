#!/bin/bash
# CELL: byte identity vs base ABOVE the floor (lane/graph-launch-guard-sweep-20260831).
# Amended A/B protocol: interleaved boot pairs x3 default (x5 only on anomaly), greedy
# (temp 0) plus seeded sampled (temp 0.7 top_p 0.9 seed 4242), 8 real agentic prompts,
# max_tokens 128. Runs on BOTH serve shapes:
#   dspark = the box12 q38+DFlash2 shape (vg pool engaged-but-inert: zero captures)
#   mtp    = the MTP-route verify-graph shape (vg pool actively replaying run_full/
#            run_segment through the guarded qwen35_verify_tparallel)
# Healthy headroom throughout; any suspended line here is a run-killing false positive.
# Usage: cell-identity-ab.sh <pair-index> <shape: dspark|mtp>
set -u
. /home/ubuntu/guard-lane/gl-lib.sh
P=${1:?pair index}
SHAPE=${2:?shape dspark|mtp}
case "$SHAPE" in
  dspark) SERVE_ENV="$SERVE_ENV_DSPARK" ;;
  mtp)    SERVE_ENV="$SERVE_ENV_MTP" ;;
  *) echo "unknown shape $SHAPE"; exit 2 ;;
esac

get_text() { python3 -c "import json,sys;d=json.load(open('$1'));t=d.get('text') or (d.get('choices') or [{}])[0].get('text') or '';sys.stdout.write(t)" 2>/dev/null; }

run_arm() { # $1=binname $2=bin
  local name=$1 bin=$2
  local LOG=serve-ab-$SHAPE-$P-$name.log
  gpu_empty || { say "REFUSING: GPU not empty"; return 2; }
  dmesg_mark
  boot "$bin" "" "$LOG" || return 1
  for i in 0 1 2 3 4 5 6 7; do
    req $i 128 0 "" $G/ab-$SHAPE-$P-$name-greedy-$i.json
  done
  for i in 0 1 2 3 4 5 6 7; do
    req $i 128 0.7 4242 $G/ab-$SHAPE-$P-$name-sampled-$i.json
  done
  local S
  S=$(suspended_count $LOG)
  if [ "${S:-0}" != "0" ]; then
    say "FALSE-POSITIVE: $S suspended line(s) at healthy headroom in $name arm; killing run"
    shutdown_srv; return 9
  fi
  shutdown_srv
  dmesg_check ab-$SHAPE-$P-$name
  return 0
}

say "=== IDENTITY PAIR $P shape=$SHAPE: base then lane (interleaved boots) ==="
run_arm base "$BINBASE" || exit $?
run_arm lane "$BINLANE" || exit $?

MISMATCH=0; EMPTY=0
for mode in greedy sampled; do
  for i in 0 1 2 3 4 5 6 7; do
    A=$(get_text $G/ab-$SHAPE-$P-base-$mode-$i.json | sha256sum | cut -d' ' -f1)
    B=$(get_text $G/ab-$SHAPE-$P-lane-$mode-$i.json | sha256sum | cut -d' ' -f1)
    AL=$(get_text $G/ab-$SHAPE-$P-base-$mode-$i.json | wc -c)
    if [ "$AL" = "0" ]; then EMPTY=$((EMPTY+1)); fi
    if [ "$A" != "$B" ]; then
      say "MISMATCH shape=$SHAPE pair=$P mode=$mode prompt=$i base=$A lane=$B"
      MISMATCH=$((MISMATCH+1))
    fi
  done
done
say "PAIR $P shape=$SHAPE VERDICT: mismatches=$MISMATCH/16 empty_base_responses=$EMPTY"
[ "$MISMATCH" = "0" ] && [ "$EMPTY" = "0" ] && exit 0 || exit 1
