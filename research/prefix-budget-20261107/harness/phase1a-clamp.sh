#!/usr/bin/env bash
# Phase 1a: spec-cache-mixed shape (c8 unique load + 6 greedy repeats, MEMRA_MAX_SESSIONS=12), five
# arms interleaved in both orders (ABCDE EDCBA ABCDE), then the cache-off reference.
#   base    main 6f5e03e74a: two-entry budget, no insert clamp, plain LRU, reclaim flush
#   lru     candidate, MEMRA_PREFIX_EVICT=lru: + live-free insert clamp, partial reclaim
#   gds     candidate (default): + value-per-byte victim order
#   flush   candidate with the reclaim flush restored (measurement-only build): clamp + GDS
#   noclamp first candidate 7b385110e4: GDS + partial reclaim, no clamp, no owed-bytes term
#           (MEMRA_ENV_AUDIT=warn: its door registry predates the MEMRA_PREFIX_EVICT row)
set -u
H=/home/avifenesh/.cache/memra-prefix-budget/harness
B=/home/avifenesh/.cache/memra-prefix-budget/bins
NEW=$B/memra-server-${NEW_LABEL:?}
FLUSH=$B/memra-server-${FLUSH_LABEL:?}
O=/home/avifenesh/.cache/memra-prefix-budget/runs/phase1a
mkdir -p "$O"
arm() { # name run
  case $1 in
    base)    bash $H/arm.sh "base-r$2" $B/memra-server-base-6f5e03e74a $O mixed MEMRA_MAX_SESSIONS=12 ;;
    lru)     bash $H/arm.sh "lru-r$2" $NEW $O mixed MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_EVICT=lru ;;
    gds)     bash $H/arm.sh "gds-r$2" $NEW $O mixed MEMRA_MAX_SESSIONS=12 ;;
    flush)   bash $H/arm.sh "flush-r$2" $FLUSH $O mixed MEMRA_MAX_SESSIONS=12 ;;
    noclamp) bash $H/arm.sh "noclamp-r$2" $B/memra-server-new-7b385110e4 $O mixed MEMRA_MAX_SESSIONS=12 MEMRA_ENV_AUDIT=warn ;;
  esac
}
r=0
for order in "base lru gds flush noclamp" "noclamp flush gds lru base" "base lru gds flush noclamp"; do
  r=$((r+1))
  for a in $order; do arm $a $r; done
done
bash $H/arm.sh off-new $NEW $O mixed MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_CACHE_MB=0
echo PHASE1A_DONE
