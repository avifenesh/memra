#!/usr/bin/env bash
# Phase 1a: spec-cache-mixed shape (c8 unique load + 6 greedy repeats, MEMRA_MAX_SESSIONS=12),
# three arms interleaved ABC CBA ABC, plus the cache-off reference on the candidate binary.
#   base = main 6f5e03e74a; lru = candidate with MEMRA_PREFIX_EVICT=lru (session-peak budget +
#   partial reclaim, LRU order); gds = candidate default.
set -u
H=/home/avifenesh/.cache/memra-prefix-budget/harness
B=/home/avifenesh/.cache/memra-prefix-budget/bins
NEW=$B/memra-server-${NEW_LABEL:?}
O=/home/avifenesh/.cache/memra-prefix-budget/runs/phase1a
mkdir -p "$O"
arm() { # name run
  case $1 in
    base) bash $H/arm.sh "base-r$2" $B/memra-server-base-6f5e03e74a $O mixed MEMRA_MAX_SESSIONS=12 ;;
    lru)  bash $H/arm.sh "lru-r$2" $NEW $O mixed MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_EVICT=lru ;;
    gds)  bash $H/arm.sh "gds-r$2" $NEW $O mixed MEMRA_MAX_SESSIONS=12 ;;
  esac
}
r=0
for order in "base lru gds" "gds lru base" "base lru gds"; do
  r=$((r+1))
  for a in $order; do arm $a $r; done
done
bash $H/arm.sh off-new $NEW $O mixed MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_CACHE_MB=0
echo PHASE1A_DONE
