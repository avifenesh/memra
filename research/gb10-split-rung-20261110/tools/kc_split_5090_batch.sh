#!/usr/bin/env bash
# Rig: kc_split_5090.sh with several splits per gpulock lease (each run from a cool card with its
# own 1 Hz thermal samples), for a shared card whose queue makes one lease per split slow.
# usage: kc_split_5090_batch.sh KERNEL_CHECK OUTDIR "S1 S2 S3" ["S4 S5 S6" ...]
set -u
KC=$1; OUT=$2; shift 2
T=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
L=$OUT/driver.log
n=0
for batch in "$@"; do
  cmds=()
  for s in $batch; do
    n=$((n+1)); tag=$(printf '%02d-sp%s' "$n" "$s")
    cmds+=("bash $T/cool5090.sh $OUT/$tag-gpu-1hz.csv env MEMRA_KC_FAST=1 MEMRA_KC_ROWS_DEPTH=262144 MEMRA_KC_ROWS_TIME=1 MEMRA_FA_SPLIT=$s timeout 10m $KC > $OUT/$tag.log 2>&1; echo \"\$(date -Is) $tag rc=\$?\" >> $L")
  done
  script=$(printf '%s\n' "${cmds[@]}")
  echo "$(date -Is) batch [$batch] start" >> "$L"
  gpulock run --ttl 40m --wait 240m --for "memra deep split rung: kernel-check timing rows at MEMRA_FA_SPLIT=[$batch] on the 5090 (one batch of short runs per lease, each from a cool card)" -- \
    bash -c "$script"
  echo "$(date -Is) batch [$batch] rc=$?" >> "$L"
done
echo "$(date -Is) all splits done" >> "$L"
