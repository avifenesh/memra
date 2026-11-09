#!/usr/bin/env bash
# memra#1093 cold-long K A/B on wrx GPU 0 (RTX PRO 6000 Blackwell Max-Q). usage: run-ab.sh SET OUT
#   SET: long (the 3.6k/5.8k set) or base (the 1.4k/3.7k set). One boot per memra-gpu-run lease.
set -u
D=/home/access/memra-1093r
export GPU_UUID=GPU-f8021f32-a83b-fa33-7ef6-03e0698a4e3b
export BOX_LOCK=/mnt/Data/work/evals/gpu0.lock
export YIELD_FILE=/mnt/Data/work/evals/gpu0.yield
export GPU_RUN=/home/access/memra-0140/repo/tools/memra-gpu-run
export CELL_MODEL=/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
export CORES=24-31,56-63
export TMPDIR=$D/tmp
mkdir -p "$TMPDIR"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,power.limit --format=csv,noheader
taskset -c "$CORES" python3 "$D/ab/ab_wrx.py" "$2" "$D/memra-server-1273a16f07" "$1"
echo "RUN_DONE $1 $(date -Is)"
