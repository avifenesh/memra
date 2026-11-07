#!/usr/bin/env bash
# Phase 1b (final candidate): agentic cells (ab_agent shape: prime pass, then the measured pass) on
# the naked server (MEMRA_CTX=32768, default caps). c8: base, lru, gds, flush interleaved ABCD DCBA;
# c4: base, gds interleaved AB BA (long runs: one run plus a validation run per arm, both orders).
set -u
H=/home/avifenesh/.cache/memra-prefix-budget/harness
B=/home/avifenesh/.cache/memra-prefix-budget/bins
NEW=$B/memra-server-${NEW_LABEL:?}
O=/home/avifenesh/.cache/memra-prefix-budget/runs/phase1b
mkdir -p "$O"
# c8 of this workload does not fit the 90 s streaming first-token deadline on 24 GB (both arms
# answered 408 in the preview), so every c8 arm runs with MEMRA_STREAM_TTFT_MS_MAX=900000: a stress
# cell for the admission reclaim, compared only across arms.
arm() { # name users run
  local x=""; [ "$2" = 8 ] && x="MEMRA_STREAM_TTFT_MS_MAX=900000"
  case $1 in
    base)  bash $H/arm.sh "c$2-base-r$3" $B/memra-server-base-6f5e03e74a $O "agentic:$2" $x ;;
    lru)   bash $H/arm.sh "c$2-lru-r$3" $NEW $O "agentic:$2" MEMRA_PREFIX_EVICT=lru $x ;;
    gds)   bash $H/arm.sh "c$2-gds-r$3" $NEW $O "agentic:$2" $x ;;
    flush) bash $H/arm.sh "c$2-flush-r$3" $B/memra-server-${FLUSH_LABEL:?} $O "agentic:$2" $x ;;
  esac
}
r=0
for order in "base lru gds flush" "flush gds lru base"; do
  r=$((r+1)); for a in $order; do arm $a 8 $r; done
done
r=0
for order in "base gds" "gds base"; do
  r=$((r+1)); for a in $order; do arm $a 4 $r; done
done
echo PHASE1B_DONE
