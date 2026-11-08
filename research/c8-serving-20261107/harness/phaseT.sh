#!/usr/bin/env bash
# Phase T (traced, equal work): agentic c8, the full shape (prime then measured pass, T=1, 768),
# MEMRA_TTFT_TRACE=1 on every arm.
#   A900: main with a 900 s first-token deadline, so it serves every turn (the work the lane serves)
#   B:    the lane as scored in phaseS8
#   B8:   the lane with the minimal-rescue prefill order and the slack-aware memory hold (CAND8)
# Order A900 B8 B B8 A900 B B8.
set -u
H=<harness>
BIN=<bins>
A=$BIN/memra-server-base-46172c1bac
B=$BIN/memra-server-ec935b6f99-s8
B8=$BIN/memra-server-${CAND8:-cand8}
O=<runs>/phaseT
mkdir -p "$O"
W=agentic:8:1.0:768
n=0
for arm in A900 B8 B B8 A900 B B8; do
  n=$((n + 1)); label=$(printf %02d-%s $n $arm)
  case $arm in
    A900) bash $H/arm2.sh "$label" "$A" "$O" $W MEMRA_TTFT_TRACE=1 MEMRA_STREAM_TTFT_MS_MAX=900000 MEMRA_SSE_PREFILL_COMMIT_MS=60000 ;;
    B) bash $H/arm2.sh "$label" "$B" "$O" $W MEMRA_TTFT_TRACE=1 ;;
    B8)
      waitfor --file <lane>/build-${CAND8:-cand8}.log --grep "^BUILD_DONE" --timeout 40m --quiet || { echo "no $B8"; exit 3; }
      bash $H/arm2.sh "$label" "$B8" "$O" $W MEMRA_TTFT_TRACE=1 ;;
  esac
done
echo PHASE_T_DONE
