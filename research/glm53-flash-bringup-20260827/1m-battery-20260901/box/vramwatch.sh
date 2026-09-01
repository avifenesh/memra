#!/usr/bin/env bash
# Per-card VRAM sampler: one CSV row per interval so the PEAK per card during a multi-minute
# prime is a banked MEASUREMENT, not a guess. The 1M prime's headroom on the tail card is the
# whole reason the 3-card resident shape is unusable, so this is a load-bearing receipt.
# Stop with a PID-verified kill of the recorded PID (echoed on start), never pkill.
set -u
OUT=$1; IV=${2:-10}
echo "vramwatch pid $$ -> $OUT every ${IV}s"
echo "ts,gpu0_mib,gpu1_mib,gpu2_mib,gpu3_mib" >> "$OUT"
while true; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -)
  echo "$(date -u +%FT%TZ),$m" >> "$OUT"
  sleep "$IV"
done
