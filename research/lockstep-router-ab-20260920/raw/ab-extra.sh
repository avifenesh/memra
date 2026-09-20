#!/usr/bin/env bash
# Residual mixed-peer dependence discriminators on the fix binary.
set -uo pipefail
source <(sed -n "/^COMMON=(/,/)$/p" /root/lane/ab.sh)
export PATH=/usr/local/cuda/bin:$PATH
ART=/data/hy3; OUT=/root/lane/ab; B=/root/lane/target-fix/release/run_lockstep
run() { local name=$1 bin=$2; shift 2; echo "== $name start $(date -u +%T)"; ( exec 9>/tmp/memra-gpu.lock; flock 9; env "${COMMON[@]}" "$@" "$bin" "$ART" ) > "$OUT/$name.log" 2>&1; echo "== $name rc=$? end $(date -u +%T)"; }
run fix-m2-mixed "$B" MEMRA_LOCKSTEP_M=2 MEMRA_NGEN=32 MEMRA_PROMPTS_FILE=$OUT/prompts.txt
run fix-m3-mixed "$B" MEMRA_LOCKSTEP_M=3 MEMRA_NGEN=32 MEMRA_PROMPTS_FILE=$OUT/prompts.txt
echo "EXTRA_DONE $(date -u +%T)"
