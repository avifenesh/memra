#!/usr/bin/env bash
# memra#1093 cost-model check on one RTX PRO 6000 Max-Q: base vs fix binaries,
# long set (base fix fix base fix base base fix), then the 1.2k set (base fix fix base).
set -u
D=/home/access/memra-1093r
export GPU_UUID=GPU-c2a4ccee-d2dd-56e8-4ea1-cc31f0190fed
export BOX_LOCK=/mnt/Data/work/evals/gpu1.lock
export YIELD_FILE=/mnt/Data/work/evals/gpu1.yield
export GPU_RUN=/home/access/memra-0140/repo/tools/memra-gpu-run
export CELL_MODEL=/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export CORES=24-31,56-63
export BIN_BASE=$D/memra-server-e0447ca00e BIN_FIX=$D/memra-server-8982282934
export TMPDIR=$D/tmp
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.limit --format=csv,noheader -i "$GPU_UUID"
python3 "$D/ab2/ab_bins.py" "$D/bins-long" long base,fix,fix,base,fix,base,base,fix
BOOT_OFFSET=1 python3 "$D/ab2/ab_bins.py" "$D/bins-base" base base,fix,fix,base
echo "RUN_DONE $(date -Is)"
