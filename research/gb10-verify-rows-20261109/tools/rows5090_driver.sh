#!/usr/bin/env bash
# Rig: a round 5 RTX 5090 B/T against main a8a980a017, B T T B B T, one gpulock lease per cell. usage: rows5090_driver.sh HEAD_SHA CAMPAIGN BASE_SHA [SET]
set -u
D=$HOME/.cache/memra-gb10-r4
CD=$D/scout/$2; mkdir -p "$CD"
B=$D/cells-work/rtx5090/bins/${3:-a8a980a017aa8541fe5ed10c63f6784c30896009}-120a
T=$D/cells-work/rtx5090/bins/$1-120a
L=$CD/driver.log
for c in ${CELLS:-01-B 02-T 03-T 04-B 05-B 06-T}; do
  bin=$B; [ "${c#*-}" = T ] && bin=$T
  echo "$(date -Is) $c start" >> "$L"
  gpulock run --ttl 15m --for "memra round 5 lever 2 (verify rows kernel) 5090 B/T ($2): $c (one cell per lease)" --wait 60m -- \
    bash "$D/box/rows5090_cell.sh" "$CD" "$c" "$bin" "${4:-28k}" >> "$L" 2>&1
  echo "$(date -Is) $c rc=$?" >> "$L"
done
echo "$(date -Is) all cells done" >> "$L"
