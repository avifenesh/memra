#!/usr/bin/env bash
# Rig: memra#1221 review S3: the band raw < required <= allocatable + pool. A 131k request on a
# fresh server; once its prime starts the holder pins MemAvailable near 3,500 MB (oom_score_adj
# 1000, 300 s cap) so the capture sees a raw reading under the requirement and MemAvailable above
# it. main's guard (bi) then the head's (vi), one lease each, none after 19:30.
set -u
H=$(cd "$(dirname "$0")" && pwd)
L=$HOME/.cache/memra-1221/logs/measure-u1.log
export NOT_AFTER=19:30
for c in "01-B bi-782dc471b7" "02-W vi-782dc471b7" "03-W vi-782dc471b7" "04-B bi-782dc471b7"; do
  set -- $c
  bash "$H/unit_cell.sh" u1 "band-$1" 15 "SIZE=131072 bash box/short_cell.sh band-u1 $1 $2 3500"; rc=$?
  echo "$(date -Is) band-$1 rc=$rc" >> "$L"
done
rsync -a --exclude='gpu.csv' --exclude='mem.jsonl' spark-u1:.cache/memra-1221/runs/band-u1/ "$HOME/.cache/memra-1221/runs/band-u1/"
echo "$(date -Is) band done" >> "$L"
