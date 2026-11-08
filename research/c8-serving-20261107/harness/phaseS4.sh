#!/usr/bin/env bash
# Phase S4 (scored): agentic c4, the full shape (prime then measured pass, T=1, 768), ABBA.
set -u
H=<harness>
BIN=<bins>
A=$BIN/memra-server-base-46172c1bac
B=$BIN/memra-server-${CAND:-e0565d0f20}
O=<runs>/phaseS4
mkdir -p "$O"
n=0
for arm in A B B A; do
  n=$((n + 1)); if [ $arm = A ]; then bin=$A; else bin=$B; fi
  bash $H/arm2.sh "$(printf %02d $n)-$arm" "$bin" "$O" agentic:4:1.0:768
done
echo PHASE_S4_DONE
