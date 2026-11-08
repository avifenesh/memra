#!/usr/bin/env bash
# Phase E: the deadline-aware prefill order engages and removes the c8 tail (diagnostic single runs).
set -u
H=<harness>
BIN=<bins>
O=<runs>/phaseE
mkdir -p "$O"
T=MEMRA_TTFT_TRACE=1
LONG="MEMRA_STREAM_TTFT_MS_MAX=900000 MEMRA_SSE_PREFILL_COMMIT_MS=60000"
bash $H/arm2.sh e1-cand3-c8 $BIN/memra-server-cand3-edf $O agentic:8:1.0:768:prime $T
bash $H/arm2.sh e2-cand3-c8-900s $BIN/memra-server-cand3-edf $O agentic:8:1.0:768:prime $T $LONG
echo PHASE_E_DONE
