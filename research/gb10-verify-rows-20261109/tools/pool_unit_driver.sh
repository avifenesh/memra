#!/usr/bin/env bash
# Rig: lever 2's pooled-cohort cell on a GB10 unit (a stable clock, unlike the laptop card): the
# 27B with its masked MTP draft as the pooled model, waves of 2, 4 and 8 concurrent requests,
# greedy and sampled, 400 tokens, main against the head, B T T B B T, one boot per sparklock lease.
# Every stream must match; the wave rates are the no-regression check (short context: every row
# is below the routing floor, so both arms launch the seqs kernel).
# usage: pool_unit_driver.sh UNIT CAMPAIGN BASE_DIR HEAD_DIR
set -u
UNIT=$1; CAMP=$2; B=$3; T=$4
D=$HOME/.cache/memra-gb10-r4
L=$D/pool-driver-$CAMP.log
G8='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF'
for arm in B1 T1 T2 B2 B3 T3; do
  bin=$B; case $arm in T*) bin=$T;; esac
  id=$(sparklock acquire --unit "$UNIT" --ttl 15m --for "memra GB10 round 5 lever 2 (verify rows kernel) pooled-cohort cell, 27B: pool-$arm (one boot per lease)" --wait 180m 2>>"$L")
  [ -n "$id" ] || { echo "$(date -Is) $arm: no lease, stopping" >> "$L"; exit 1; }
  echo "$(date -Is) $arm: lease $id" >> "$L"
  /usr/bin/ssh -n "spark-$UNIT" "cd ~/.cache/memra-gb10-r4 && mkdir -p runs/$CAMP && POOL_MODELS=q9=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf MEMRA_MTP_DRAFT=$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf MEMRA_PREFIX_STABLE_BOUNDARY=1 POOL_MAX_TOKENS=400 timeout 13m python3 box/pool_cell.py \$HOME/.cache/memra-gb10-r4/bins/$bin/memra-server \$HOME/.cache/memra-gb10-r4/runs/$CAMP/pool-$arm 8351" >> "$L" 2>&1
  echo "$(date -Is) $arm: rc=$?" >> "$L"
  sparklock release "$id" --unit "$UNIT" >> "$L" 2>&1
done
echo "$(date -Is) all cells done" >> "$L"
