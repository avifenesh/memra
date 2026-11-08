#!/usr/bin/env bash
# Phase E3: cand6 (unconditional hold, at-risk prefill order, seed batching, headroom yield).
set -u
H=<harness>
BIN=<bins>
bash $H/arm2.sh e5-cand6-c8 $BIN/memra-server-cand6 <runs>/phaseE agentic:8:1.0:768:prime MEMRA_TTFT_TRACE=1
bash $H/phaseX.sh x6-cand6-c8-g1 $BIN/memra-server-cand6 8
echo PHASE_E3_DONE
