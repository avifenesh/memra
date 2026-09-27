#!/usr/bin/env bash
# tools/kv-host-contract-fault-gate.sh rerun on the same day-16 binary after the gate's own matcher fix
# (75573cad3: after_any for the promote cells). usage: faultgate-fix-cell.sh <lockfd> <cache_mb>
set -uo pipefail
fd=$1; cache_mb=$2
R=/root/spill-receipts/c-day16
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
git rev-parse HEAD
sha256sum $R/bins/memra-server
env MEMRA_GPU_LOCK=/tmp/memra-gpu.lock "MEMRA_HOSTGATE_CACHE_MB=$cache_mb" CUDA_VISIBLE_DEVICES=0 bash tools/kv-host-contract-fault-gate.sh --external-lock "$fd" /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf $R/bins/memra-server $R/faultgate-fix/ev
