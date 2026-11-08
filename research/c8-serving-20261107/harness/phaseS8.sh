#!/usr/bin/env bash
# Phase S8 (scored): agentic c8, the full shape (prime pass then the measured pass, T=1, 768 new
# tokens, naked defaults), main (A) against the candidate (B), fresh boot per run, ABBAAB.
set -u
H=<harness>
BIN=<bins>
A=$BIN/memra-server-base-46172c1bac
B=$BIN/memra-server-${CAND:-cand3-edf}
O=<runs>/phaseS8
mkdir -p "$O"
n=0
for arm in A B B A A B; do
  n=$((n + 1))
  if [ $arm = A ]; then bin=$A; else bin=$B; fi
  waitfor --file "$bin" --timeout 30m --quiet || { echo "no binary $bin"; exit 3; }
  bash $H/arm2.sh "$(printf %02d $n)-$arm" "$bin" "$O" agentic:8:1.0:768
done
echo PHASE_S8_DONE
