#!/usr/bin/env bash
# kv-host-spill-{identity,failure}-gate on the day-15 binary under the inherited collector lock.
# usage: hostgate-cell.sh <identity|failure> <off|on> <spec:default|plain> <lockfd> <cache_mb>
set -uo pipefail
gate=$1; arm=$2; spec=$3; fd=$4; cache_mb=$5
R=/root/spill-receipts/c-day15
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
env_extra=(MEMRA_GPU_LOCK=/tmp/memra-gpu.lock "MEMRA_HOSTGATE_CACHE_MB=$cache_mb" CUDA_VISIBLE_DEVICES=0)
[ "$arm" = on ] && env_extra+=(MEMRA_KV_HOST_CONTRACTS=1)
[ "$spec" = plain ] && env_extra+=(MEMRA_SERVE_SPEC=0)
sha256sum $R/bins/memra-server
env "${env_extra[@]}" bash tools/kv-host-spill-$gate-gate.sh --external-lock "$fd" /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf $R/bins/memra-server $R/hostgate-$gate-$arm-$spec/ev
