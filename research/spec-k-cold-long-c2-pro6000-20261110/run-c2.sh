#!/usr/bin/env bash
# memra#1093: two requests at once on one RTX PRO 6000 Max-Q, the flip PR's parent (base) against
# its head (flip), 12 boots base flip flip base x3, one boot per lease.
# usage: run-c2.sh BASE_COMMIT FLIP_COMMIT DEADLINE_HHMM
set -u
D=/home/access/memra-1093r
export GPU_UUID=GPU-c2a4ccee-d2dd-56e8-4ea1-cc31f0190fed
export BOX_LOCK=/mnt/Data/work/evals/gpu1.lock
export YIELD_FILE=/mnt/Data/work/evals/gpu1.yield
export GPU_RUN=/home/access/memra-0140/repo/tools/memra-gpu-run
export CELL_MODEL=/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export CORES=24-31,56-63
export BIN_BASE=$D/memra-server-$1 BIN_FLIP=$D/memra-server-$2
export DEADLINE=$3
export TMPDIR=$D/tmp
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.limit --format=csv,noheader -i "$GPU_UUID"
CELL_SCRIPT=c2p_ab.py python3 "$D/ab3/ab_bins.py" "$D/c2" c2 base,flip,flip,base,base,flip,flip,base,base,flip,flip,base
echo "RUN_DONE $(date -Is)"
