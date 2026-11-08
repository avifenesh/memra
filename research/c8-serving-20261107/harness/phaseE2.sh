#!/usr/bin/env bash
# Phase E2: cand4 (hold and prefill order only when a deadline is at risk) at c8, prime pass.
set -u
H=<harness>
BIN=<bins>
O=<runs>/phaseE
T=MEMRA_TTFT_TRACE=1
LONG="MEMRA_STREAM_TTFT_MS_MAX=900000 MEMRA_SSE_PREFILL_COMMIT_MS=60000"
bash $H/arm2.sh e3-cand4-c8 $BIN/memra-server-cand4-risk $O agentic:8:1.0:768:prime $T
bash $H/arm2.sh e4-cand4-c8-900s $BIN/memra-server-cand4-risk $O agentic:8:1.0:768:prime $T $LONG
echo PHASE_E2_DONE
