#!/usr/bin/env bash
# tools/kv-host-contract-fault-gate.sh (door ON, one-shot contract-presubmit and contract-postpublish faults)
# on the review binary under the inherited collector lock. usage: faultgate-cell.sh <lockfd> <cache_mb>
set -uo pipefail
fd=$1; cache_mb=$2
R=/root/spill-receipts/c-day15-review
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
sha256sum $R/bins/memra-server
env MEMRA_GPU_LOCK=/tmp/memra-gpu.lock "MEMRA_HOSTGATE_CACHE_MB=$cache_mb" CUDA_VISIBLE_DEVICES=0 bash tools/kv-host-contract-fault-gate.sh --external-lock "$fd" /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf $R/bins/memra-server $R/faultgate/ev
