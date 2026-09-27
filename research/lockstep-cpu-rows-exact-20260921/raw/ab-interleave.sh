#!/usr/bin/env bash
# Default decision A/B for the lockstep CPU rows arm: A = exact multi-row (MEMRA_LOCKSTEP_CPU_ROWS=1), B = one job per row (=0).
# Same binary (lane), same box, same window, M=4 mixed prompts, 64 new tokens; 5 AB pairs then 5 BA pairs.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH
source <(sed -n "/^COMMON=(/,/)$/p" /root/lane/ab-rows.sh)
ART=/data/hy3; OUT=/root/lane/abint; mkdir -p "$OUT"; cp /root/lane/abrows/prompts.txt "$OUT/"
LANE=/root/lane/target-lane/release/run_lockstep
one() { local arm=$1 tag=$2; local rows=1; [ "$arm" = B ] && rows=0
  ( exec 9>/tmp/memra-gpu.lock; flock 9; env "${COMMON[@]}" MEMRA_NGEN=64 MEMRA_LOCKSTEP_M=4 MEMRA_PROMPTS_FILE="$OUT/prompts.txt" MEMRA_LOCKSTEP_CPU_ROWS=$rows "$LANE" "$ART" ) > "$OUT/$tag-$arm.log" 2>&1
  echo "$tag $arm $(grep -h "lockstep m=" "$OUT/$tag-$arm.log" | grep -oE "= [0-9.]+ tok/s aggregate" | grep -oE "[0-9.]+") $(date -u +%T)" | tee -a "$OUT/results.tsv"
}
: > "$OUT/results.tsv"
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader > "$OUT/gpu-before.txt"
for i in 1 2 3 4 5; do one A ab$i; one B ab$i; done
for i in 1 2 3 4 5; do one B ba$i; one A ba$i; done
nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm --format=csv,noheader > "$OUT/gpu-after.txt"
echo "ABINT_DONE $(date -u +%T)"
