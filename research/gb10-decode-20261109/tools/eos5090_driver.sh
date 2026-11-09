#!/usr/bin/env bash
# Rig: round 5 lever 1's RTX 5090 B/T, B T T B B T, one gpulock lease per cell.
set -u
D=~/.cache/memra-gb10-r4
CD=$D/scout/eos-5090; mkdir -p "$CD"
B=$D/cells-work/rtx5090/bins/a8a980a017aa8541fe5ed10c63f6784c30896009-120a
T=$D/cells-work/rtx5090/bins/$1-120a
L=$CD/driver.log
for c in 01-B 02-T 03-T 04-B 05-B 06-T; do
  bin=$B; [ "${c#*-}" = T ] && bin=$T
  echo "$(date -Is) $c start" >> "$L"
  gpulock run --ttl 15m --for "memra round 5 lever 1 (EOS ends the spec burst) 5090 B/T: $c (one cell per lease)" --wait 60m -- \
    bash "$D/box/eos5090_cell.sh" "$CD" "$c" "$bin" >> "$L" 2>&1
  echo "$(date -Is) $c rc=$?" >> "$L"
done
echo "$(date -Is) all cells done" >> "$L"
