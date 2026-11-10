#!/usr/bin/env bash
# Rig: the deep rung's RTX 5090 served B/T, B T T B B T, one gpulock lease per cell (landing
# waiters first at every boundary), each cell from a cool card with 1 Hz thermal samples
# (cool5090.sh). A failed cell stops the driver.
# Each cell starts only after no landing lease or ticket has shown for 5 minutes (landing_quiet.sh).
# usage: split5090_driver.sh CAMPAIGN_DIR BASE_BIN_DIR HEAD_BIN_DIR SET
set -u
CD=$1; B=$2; T=$3; SET=$4
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$CD"
L=$CD/driver.log
for c in ${CELLS:-01-B 02-T 03-T 04-B 05-B 06-T}; do
  bin=$B; [ "${c#*-}" = T ] && bin=$T
  echo "$(date -Is) $c start" >> "$L"
  waitfor --until "bash $(dirname "$0")/landing_quiet.sh" --every 20s --timeout 180m --quiet >> "$L" 2>&1
  gpulock run --ttl 20m --wait 300m --for "memra deep split rung: 5090 served B/T, $SET set, $c (one boot per lease, cool start)" -- \
    bash "$H/cool5090.sh" "$CD/$c-gpu-1hz.csv" bash "$H/split5090_cell.sh" "$CD" "$c" "$bin" "$SET" >> "$L" 2>&1
  echo "$(date -Is) $c rc=$?" >> "$L"
  grep -q "SPLIT-CELL-DONE $c" "$CD/campaign.log" 2>/dev/null || { echo "$(date -Is) $c FAILED, stopping" >> "$L"; exit 1; }
done
echo "$(date -Is) all cells done" >> "$L"
