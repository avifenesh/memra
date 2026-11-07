#!/usr/bin/env bash
# Phase 1c: c1 agentic (two runs per arm, ABC CBA), then greedy exactness at c1 (T=0, 256 new
# tokens) against the cache-off boot of the candidate binary.
set -u
H=/home/avifenesh/.cache/memra-prefix-budget/harness
B=/home/avifenesh/.cache/memra-prefix-budget/bins
NEW=$B/memra-server-${NEW_LABEL:?}
BASE=$B/memra-server-base-6f5e03e74a
O=/home/avifenesh/.cache/memra-prefix-budget/runs/phase1c
mkdir -p "$O"
r=0
for order in "base lru gds" "gds lru base"; do
  r=$((r+1))
  for a in $order; do
    case $a in
      base) bash $H/arm.sh "c1-base-r$r" $BASE $O agentic:1 ;;
      lru)  bash $H/arm.sh "c1-lru-r$r" $NEW $O agentic:1 MEMRA_PREFIX_EVICT=lru ;;
      gds)  bash $H/arm.sh "c1-gds-r$r" $NEW $O agentic:1 ;;
    esac
  done
done
bash $H/arm.sh exact-off $NEW $O agentic:1:0:256 MEMRA_PREFIX_CACHE_MB=0
bash $H/arm.sh exact-gds $NEW $O agentic:1:0:256
bash $H/arm.sh exact-lru $NEW $O agentic:1:0:256 MEMRA_PREFIX_EVICT=lru
bash $H/arm.sh exact-base $BASE $O agentic:1:0:256
echo PHASE1C_DONE
