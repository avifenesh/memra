#!/usr/bin/env bash
# Phase S1 (no-op check): agentic c1, the full shape, AB. Neither change can engage at c1: the hold
# needs a memory defer, which needs a peer; the seed batching and the deadline order need two rows.
set -u
H=<harness>
BIN=<bins>
A=$BIN/memra-server-base-46172c1bac
B=$BIN/memra-server-${CAND:-e0565d0f20}
O=<runs>/phaseS1
mkdir -p "$O"
bash $H/arm2.sh 01-A "$A" "$O" agentic:1:1.0:768
bash $H/arm2.sh 02-B "$B" "$O" agentic:1:1.0:768
echo PHASE_S1_DONE
