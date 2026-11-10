#!/usr/bin/env bash
# Rig: kernel-check's FA-VERIFY-ROWS timing rows on the RTX 5090 under MEMRA_FA_SPLIT=S, one
# gpulock lease per split, each from a cool card with 1 Hz thermal samples (cool5090.sh).
# The first split is repeated at the end as the drift check.
# usage: kc_split_5090.sh KERNEL_CHECK OUTDIR [SPLITS...]
set -u
KC=$1; OUT=$2; shift 2
SPLITS=${*:-128 256 512 1024 128}
T=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
L=$OUT/driver.log
n=0
for s in $SPLITS; do
  n=$((n+1)); tag=$(printf '%02d-sp%s' "$n" "$s")
  echo "$(date -Is) $tag start" >> "$L"
  gpulock run --ttl 25m --wait 180m --for "memra deep split rung: kernel-check timing rows at MEMRA_FA_SPLIT=$s on the 5090 (one split per lease)" -- \
    bash "$T/cool5090.sh" "$OUT/$tag-gpu-1hz.csv" \
    env MEMRA_KC_FAST=1 MEMRA_KC_ROWS_DEPTH=262144 MEMRA_KC_ROWS_TIME=1 MEMRA_FA_SPLIT="$s" \
    timeout 22m "$KC" > "$OUT/$tag.log" 2>&1
  echo "$(date -Is) $tag rc=$?" >> "$L"
done
echo "$(date -Is) all splits done" >> "$L"
