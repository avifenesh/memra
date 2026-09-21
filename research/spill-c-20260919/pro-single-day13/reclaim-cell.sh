#!/usr/bin/env bash
# Lane B prefix-evict-reclaim gate (tools copy at its lane tip, sha256 in tools.sha256) on the day-13 binary.
# usage: reclaim-cell.sh <off|on> <lockfd>
set -uo pipefail
arm=$1; fd=$2
R=/root/spill-receipts/c-day13
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
env_extra=(CUDA_VISIBLE_DEVICES=0)
[ "$arm" = on ] && env_extra+=(MEMRA_KV_HOST_CONTRACTS=1)
sha256sum $R/bins/memra-server
env "${env_extra[@]}" python3 $R/tools/prefix-evict-reclaim-gate.py --external-lock "$fd" --model /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf --bin $R/bins/memra-server --out $R/reclaim-$arm/cell
