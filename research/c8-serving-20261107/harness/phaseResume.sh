#!/usr/bin/env bash
# Resume of the c4/c1/mixed block after the 12:46 yield: c4 runs 04-A, 05-A, 06-B (the phase
# becomes A B B A A B; 01-03 ran before the yield), then c1 AB and the mixed shape ABBAAB plus the
# cache-off references. The candidate is the final code on the measured base.
set -u
H=<harness>
BIN=<bins>
A=$BIN/memra-server-base-46172c1bac
B=$BIN/memra-server-ec935b6f99-s8
O=<runs>/phaseS4
echo "S4 resume $(date -Is)"
bash $H/arm2.sh 04-A "$A" "$O" agentic:4:1.0:768
bash $H/arm2.sh 05-A "$A" "$O" agentic:4:1.0:768
bash $H/arm2.sh 06-B "$B" "$O" agentic:4:1.0:768
echo "S1 start $(date -Is)"; CAND=ec935b6f99-s8 bash $H/phaseS1.sh
echo "SM start $(date -Is)"; CAND=ec935b6f99-s8 bash $H/phaseSM.sh
echo "RESUME_DONE $(date -Is)"
