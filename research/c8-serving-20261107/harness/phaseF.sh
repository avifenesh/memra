#!/usr/bin/env bash
# The second scored block, one lease: the greedy c8 exactness cell on the final tree, then c4 ABBA,
# c1 AB and the mixed shape ABBAAB plus cache-off references. CAND names the candidate binary.
set -u
H=<harness>
BIN=<bins>
: "${CAND:?CAND names the candidate binary suffix}"
test -x "$BIN/memra-server-$CAND" || { echo "no candidate binary $CAND"; exit 2; }
echo "X start $(date -Is)"
bash $H/phaseX.sh "x7-$CAND-c8-g1" "$BIN/memra-server-$CAND" 8
echo "S4 start $(date -Is)"
bash $H/phaseS4.sh
echo "S1 start $(date -Is)"
bash $H/phaseS1.sh
echo "SM start $(date -Is)"
bash $H/phaseSM.sh
echo "PHASE_F_DONE $(date -Is)"
