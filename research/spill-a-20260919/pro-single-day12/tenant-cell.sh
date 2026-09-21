#!/usr/bin/env bash
# kv-host-tenant-reclaim-gate (memra#384) on one arm's binary under the inherited collector lock.
# The gate script is the lane's (/root/wt-a on lane-a-day12); only the binary differs per arm.
# usage: tenant-cell.sh <base|fix> <lockfd> <cell-name> [binary-suffix, default = arm]
set -uo pipefail
arm=$1; fd=$2; cell=$3; bin=${4:-$arm}
R=/root/spill-receipts/a-day12
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
git rev-parse HEAD
sha256sum "$R/bins/memra-server-$bin"
env MEMRA_GPU_LOCK=/tmp/memra-gpu.lock CUDA_VISIBLE_DEVICES=0 MEMRA_GATE_PORT=18129 \
  bash tools/kv-host-tenant-reclaim-gate.sh --external-lock "$fd" "$arm" \
  /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf "$R/bins/memra-server-$bin" "$R/$cell/ev"
