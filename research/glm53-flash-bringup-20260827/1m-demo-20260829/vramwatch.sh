#!/bin/bash
# Per-card VRAM sampler for the 1M demonstration: appends one CSV row per interval so the
# PEAK per card during a multi-hour prime is a banked measurement, not a guess. Stop with
# a PID-verified kill of the recorded PID (echoed on start), never pkill.
# usage: vramwatch.sh <out.csv> [interval_s=10]
set -u
OUT=$1; IV=${2:-10}
echo "vramwatch pid $$ -> $OUT every ${IV}s"
echo "ts,gpu0_mib,gpu1_mib,gpu2_mib,gpu3_mib" >> "$OUT"
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)
  echo "$(date -u +%FT%TZ),$m" >> "$OUT"
  sleep "$IV"
done
