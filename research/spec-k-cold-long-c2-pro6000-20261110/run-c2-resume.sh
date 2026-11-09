#!/usr/bin/env bash
# memra#1093: boots 9 to 12 of the two-at-once cell (run-c2.sh), after the box's quiet window
# paused it at a boot boundary from 00:26 to 00:33.
# usage: run-c2-resume.sh BASE_COMMIT FLIP_COMMIT DEADLINE_HHMM
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
mv "$D/c2/YIELD" "$D/c2/YIELD-quiet-window-0026-0033"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.limit --format=csv,noheader -i "$GPU_UUID"
CELL_SCRIPT=c2p_ab.py BOOT_OFFSET=9 python3 "$D/ab3/ab_bins.py" "$D/c2" c2 base,flip,flip,base
echo "RUN_DONE $(date -Is)"
