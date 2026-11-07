#!/usr/bin/env bash
# Phase 1c: c1 agentic (base and gds, two runs per arm, AB BA), then greedy exactness at c1 (T=0, 256 new
# tokens) against the cache-off boot of the candidate binary.
set -u
H=<harness>
B=<bins>
NEW=$B/memra-server-${NEW_LABEL:?}
BASE=$B/memra-server-base-6f5e03e74a
O=<runs>/phase1c
mkdir -p "$O"
r=0
for order in "base gds" "gds base"; do
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
bash $H/arm.sh exact-base $BASE $O agentic:1:0:256
echo PHASE1C_DONE
