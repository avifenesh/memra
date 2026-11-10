#!/usr/bin/env bash
# Rig: kernel-check's FA-VERIFY-ROWS timing rows on a GB10 unit under MEMRA_FA_SPLIT=S, one
# sparklock lease per split (a landing waiter goes first at every boundary). The first split is
# repeated at the end as the drift check.
# usage: kc_split_unit.sh UNIT BIN_NAME OUTDIR [SPLITS...]
#   BIN_NAME: the directory under ~/.cache/memra-split-rung/bins on the unit holding kernel-check
set -u
UNIT=$1; BN=$2; OUT=$3; shift 3
SPLITS=${*:-128 256 512 1024 128}
mkdir -p "$OUT"
L=$OUT/driver.log
n=0
for s in $SPLITS; do
  n=$((n+1)); tag=$(printf '%02d-sp%s' "$n" "$s")
  id=$(sparklock acquire --unit "$UNIT" --ttl 25m --for "memra deep split rung: kernel-check timing rows at MEMRA_FA_SPLIT=$s (one split per lease)" --wait 180m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $tag: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) $tag: lease $id" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "cd ~/.cache/memra-split-rung && nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader && MEMRA_KC_FAST=1 MEMRA_KC_ROWS_DEPTH=262144 MEMRA_KC_ROWS_TIME=1 MEMRA_FA_SPLIT=$s timeout 22m bins/$BN/kernel-check && nvidia-smi --query-gpu=temperature.gpu,clocks.sm --format=csv,noheader" > "$OUT/$tag.log" 2>&1
  echo "$(date -Is) $tag: rc=$?" >> "$L"
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
  echo "$(date -Is) $tag: released" >> "$L"
done
echo "$(date -Is) all splits done" >> "$L"
