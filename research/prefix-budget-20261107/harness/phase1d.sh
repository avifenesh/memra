#!/usr/bin/env bash
# Phase 1d: the PREFIX-CACHE-POLICY incident replay (day-20 5090 geometry), lru vs gds,
# AB/BA x 5 pairs, digest identity against a cache-off boot. Runs under `gpulock run`, which
# hands the canonical 5090 lock on FD 9.
set -u
H=<harness>
B=<bins>
NEW=$B/memra-server-${NEW_LABEL:?}
O=<runs>/phase1d
mkdir -p "$O"
MEMRA_REUSE_POOL=0 python3 $H/prefix-evict-ab.py --external-lock 9 \
  --model <models>/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf --bin "$NEW" --out "$O/${CELL:-cell}" \
  --budget-mib 1024 --cohort-tokens 1248,1344,1440,1536 \
  --turns 12 --start-tokens 10912 --grow-tokens 160 --return-every 3 --pairs "${PAIRS:-5}" \
  --ctx 16384 --max-tokens 8 --gpu-lock /tmp/memra-5090.lock > "$O/${CELL:-cell}-driver.log" 2>&1
echo "rc=$?" >> "$O/${CELL:-cell}-driver.log"
echo PHASE1D_DONE
