#!/usr/bin/env bash
# Phase R: reproduce on main (46172c1bac). Run under one gpulock lease.
#   R1 agentic c8 prime pass, naked defaults (cache on, 90 s first-token deadline)
#   R2 the same, prefix cache off
#   R3 the same, cache on, 900 s first-token deadline (where does TTFT land without the deadline?)
#   R4 agentic c1 prime pass, cache off (cold prefill rate per prompt)
#   R5 mixed shape, cache on;  R6 mixed shape, cache off
# Every arm runs with MEMRA_TTFT_TRACE=1 (one [ttft] phase line per request).
set -u
H=<harness>
B=<bins>/memra-server-base-46172c1bac
O=<runs>/phaseR
mkdir -p "$O"
T=MEMRA_TTFT_TRACE=1
bash $H/arm.sh r1-c8-on $B $O agentic:8:1.0:768:prime $T
bash $H/arm.sh r2-c8-off $B $O agentic:8:1.0:768:prime $T MEMRA_PREFIX_CACHE_MB=0
bash $H/arm.sh r3-c8-on-900s $B $O agentic:8:1.0:768:prime $T MEMRA_STREAM_TTFT_MS_MAX=900000
bash $H/arm.sh r4-c1-off $B $O agentic:1:1.0:768:prime $T MEMRA_PREFIX_CACHE_MB=0
bash $H/arm.sh r5-mixed-on $B $O mixed $T MEMRA_MAX_SESSIONS=12
bash $H/arm.sh r6-mixed-off $B $O mixed $T MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_CACHE_MB=0
echo PHASE_R_DONE
