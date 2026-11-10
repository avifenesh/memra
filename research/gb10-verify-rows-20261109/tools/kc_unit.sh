#!/usr/bin/env bash
# Rig: lever 2's kernel-check cell on a GB10 unit under one sparklock lease: the synthetic sections
# (MEMRA_KC_FAST=1) with the FA-VERIFY-ROWS timing rows. usage: kc_unit.sh UNIT SHA10 OUTDIR
set -u
UNIT=$1; H=$2; OUT=$3
L=$OUT/driver.log
id=$(sparklock acquire --unit "$UNIT" --ttl 20m --for "memra GB10 round 5 lever 2 (verify rows kernel): kernel-check FA-VERIFY-ROWS bit identity + per-call time (one cell per lease)" --wait 60m 2>>"$L")
[ -n "$id" ] || { echo "$(date -Is) no lease" >> "$L"; exit 1; }
echo "$(date -Is) lease $id" >> "$L"
/usr/bin/ssh -n "spark-$UNIT" "cd ~/.cache/memra-gb10-r4 && MEMRA_KC_FAST=1 MEMRA_KC_ROWS_DEPTH=262144 MEMRA_KC_ROWS_TIME=1 timeout 18m bins/rows-$H/kernel-check" > "$OUT/kc.log" 2>&1
echo "$(date -Is) rc=$?" >> "$L"
sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
echo "$(date -Is) released" >> "$L"
