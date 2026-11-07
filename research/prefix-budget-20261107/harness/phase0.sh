#!/usr/bin/env bash
# Phase 0: baseline numbers on main (mixed shape), and the cache-off reference for exactness.
set -u
H=/home/avifenesh/.cache/memra-prefix-budget/harness
B=/home/avifenesh/.cache/memra-prefix-budget/bins
O=/home/avifenesh/.cache/memra-prefix-budget/runs/phase0
mkdir -p "$O"
bash $H/arm.sh base-mixed $B/memra-server-base-6f5e03e74a $O mixed MEMRA_MAX_SESSIONS=12
bash $H/arm.sh off-mixed $B/memra-server-base-6f5e03e74a $O mixed MEMRA_MAX_SESSIONS=12 MEMRA_PREFIX_CACHE_MB=0
echo PHASE0_DONE
