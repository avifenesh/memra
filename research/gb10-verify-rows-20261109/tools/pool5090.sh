#!/usr/bin/env bash
# Rig: lever 2's pooled-cohort cell on the RTX 5090: research/gb10-decode-20261109/tools/pool_cell.py
# (q9, waves of 2, 4 and 8 concurrent requests, greedy and sampled, POOL_MAX_TOKENS=400) on main and
# on the head, B T T B B T, one boot per gpulock lease, each boot from a cool card (cool5090.sh,
# 1 Hz thermal samples). Every stream must match; the wave rates are the no-regression check.
# usage: pool5090.sh CAMPAIGN_DIR BASE_BIN_DIR HEAD_BIN_DIR
set -u
export PATH=/usr/bin:$PATH
CD=$1; B=$2; T=$3
R=$HOME/.cache/memra-gb10-r5/rows
P=$HOME/.cache/memra-gb10-r4/wt-eos/research/gb10-decode-20261109/tools/pool_cell.py
mkdir -p "$CD"
for arm in B1 T1 T2 B2 B3 T3; do
  bin=$B; case $arm in T*) bin=$T;; esac
  echo "$(date -Is) pool-$arm start" >> "$CD/driver.log"
  gpulock run --ttl 20m --for "memra round 5 lever 2 (verify rows kernel, routed) pooled-cohort cell on the 5090, cool start: pool-$arm (one boot per lease)" --wait 120m -- \
    bash "$R/cool5090.sh" "$CD/gpu-$arm.csv" env POOL_MAX_TOKENS=400 /usr/bin/python3 "$P" "$bin/memra-server" "$CD/pool-$arm" 8351 >> "$CD/driver.log" 2>&1
  echo "$(date -Is) pool-$arm rc=$? $(tail -n 1 "$CD/gpu-$arm.csv")" >> "$CD/driver.log"
done
echo "$(date -Is) all cells done" >> "$CD/driver.log"
