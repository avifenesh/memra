#!/usr/bin/env bash
# Rig: after u2's first chain (the BT/ST builds), copy BT and ST to u1 (sha checked), then run
# u1's second chain after its first. usage: u1_next.sh CB33_CHAIN_PID B23D_CHAIN_PID
set -u
R=$HOME/.cache/memra-1221b; L=$R/logs/u1-next.log
waitfor --gone "$1" --timeout 120m --quiet || { echo "$(date -Is) u2 chain still alive" >> "$L"; exit 1; }
for d in BT-dae162ddef ST-S-6fd1564e74; do
  mkdir -p "$R/xfer/$d"
  rsync -a "spark-u2:.cache/memra-1221b/bins/$d/memra-server" "$R/xfer/$d/" && rsync -a "$R/xfer/$d" spark-u1:.cache/memra-1221b/bins/ || { echo "$(date -Is) copy $d failed" >> "$L"; exit 1; }
  a=$(sha256sum "$R/xfer/$d/memra-server" | cut -c1-64); b=$(ssh -n spark-u1 "sha256sum ~/.cache/memra-1221b/bins/$d/memra-server" | cut -c1-64)
  echo "$(date -Is) $d $a u1 $b" >> "$L"; [ "$a" = "$b" ] || exit 1
done
WAITPID=$2 bash "$R/tools/chain.sh" u1 "$R/logs/steps-u1-2.txt"
