#!/usr/bin/env bash
# Phase H: attribution single runs (diagnostic, not scored).
#   r3b  main, agentic c8 prime pass, cache on, 900 s first-token deadline (+ the 60 s SSE commit it needs)
#   h1   head-of-line only, agentic c8 prime pass, naked defaults
#   h2   head-of-line only, the r3b shape (where TTFT lands without the deadline)
#   s1   seed-batch only, mixed shape, cache on
#   b1   both, mixed shape, cache on
#   b2   both, agentic c8 prime pass, naked defaults
set -u
H=<harness>
BIN=<bins>
O=<runs>/phaseH
mkdir -p "$O"
T=MEMRA_TTFT_TRACE=1
LONG="MEMRA_STREAM_TTFT_MS_MAX=900000 MEMRA_SSE_PREFILL_COMMIT_MS=60000"
bash $H/arm2.sh r3b-base-c8-900s $BIN/memra-server-base-46172c1bac $O agentic:8:1.0:768:prime $T $LONG
bash $H/arm2.sh h1-hol-c8 $BIN/memra-server-hol-only $O agentic:8:1.0:768:prime $T
bash $H/arm2.sh h2-hol-c8-900s $BIN/memra-server-hol-only $O agentic:8:1.0:768:prime $T $LONG
bash $H/arm2.sh s1-seed-mixed $BIN/memra-server-cand1-seedbatch $O mixed $T MEMRA_MAX_SESSIONS=12
bash $H/arm2.sh b1-both-mixed $BIN/memra-server-cand2-both $O mixed $T MEMRA_MAX_SESSIONS=12
bash $H/arm2.sh b2-both-c8 $BIN/memra-server-cand2-both $O agentic:8:1.0:768:prime $T
echo PHASE_H_DONE
