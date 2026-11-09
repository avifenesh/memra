#!/usr/bin/env bash
# memra#1093 check of the round-rate sample candidate on one RTX PRO 6000 Max-Q: the base,
# the flip binary with the cold-long rung forced back to 3, and the flip binary at its default,
# one request at a time; then base against flip with four requests at once.
set -u
D=/home/access/memra-1093r
export GPU_UUID=GPU-c2a4ccee-d2dd-56e8-4ea1-cc31f0190fed
export BOX_LOCK=/mnt/Data/work/evals/gpu1.lock
export YIELD_FILE=/mnt/Data/work/evals/gpu1.yield
export GPU_RUN=/home/access/memra-0140/repo/tools/memra-gpu-run
export CELL_MODEL=/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export CORES=24-31,56-63
export BIN_BASE=$D/memra-server-e0447ca00e BIN_FLIP=$D/memra-server-$1
export DEADLINE=22:52
export TMPDIR=$D/tmp
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.limit --format=csv,noheader -i "$GPU_UUID"
python3 "$D/ab3/ab_bins.py" "$D/sample-long" long rung3,base,flip,flip,base,rung3,rung3,base,flip
CELL_SCRIPT=c4_ab.py python3 "$D/ab3/ab_bins.py" "$D/sample-c4" c4 base,flip,flip,base
echo "RUN_DONE $(date -Is)"
