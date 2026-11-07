#!/usr/bin/env bash
# Phase 1b: agentic cells (ab_agent shape: prime pass, then the measured pass) on the naked
# server (MEMRA_CTX=32768, default caps), three arms interleaved ABC CBA at c4 and c8 (long
# runs: one run plus a validation run per arm, both orders).
set -u
H=/home/avifenesh/.cache/memra-prefix-budget/harness
B=/home/avifenesh/.cache/memra-prefix-budget/bins
NEW=$B/memra-server-${NEW_LABEL:?}
O=/home/avifenesh/.cache/memra-prefix-budget/runs/phase1b
mkdir -p "$O"
arm() { # name users run
  case $1 in
    base) bash $H/arm.sh "c$2-base-r$3" $B/memra-server-base-6f5e03e74a $O "agentic:$2" ;;
    lru)  bash $H/arm.sh "c$2-lru-r$3" $NEW $O "agentic:$2" MEMRA_PREFIX_EVICT=lru ;;
    gds)  bash $H/arm.sh "c$2-gds-r$3" $NEW $O "agentic:$2" ;;
  esac
}
for u in ${USERS_LIST:-4 8}; do
  r=0
  for order in "base lru gds" "gds lru base"; do
    r=$((r+1))
    for a in $order; do arm $a $u $r; done
  done
done
echo PHASE1B_DONE
