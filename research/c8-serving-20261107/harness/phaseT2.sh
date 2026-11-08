#!/usr/bin/env bash
# Phase T2 (traced): agentic c8, the full shape, the reviewed refinement (B10) against the lane as
# scored in phaseS8 (B), B10 B B10 B B10, MEMRA_TTFT_TRACE=1 on both arms.
set -u
H=<harness>
BIN=<bins>
B=$BIN/memra-server-ec935b6f99-s8
B10=$BIN/memra-server-${CAND10:-cand10}
O=<runs>/phaseT2
mkdir -p "$O"
test -x "$B10" || { echo "no $B10"; exit 3; }
n=0
for arm in B10 B B10 B B10; do
  n=$((n + 1)); label=$(printf %02d-%s $n $arm)
  case $arm in
    B) bin=$B ;;
    B10) bin=$B10 ;;
  esac
  bash $H/arm2.sh "$label" "$bin" "$O" agentic:8:1.0:768 MEMRA_TTFT_TRACE=1
done
echo PHASE_T2_DONE
