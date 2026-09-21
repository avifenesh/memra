#!/usr/bin/env bash
# lane A's kv-host-tenant-reclaim-gate, fix arm, on the day-15 binary; arm off|on sets the door.
# The gate sets its own tier env (host 1024 MiB, tenant share 38%, device 384 MiB, two keyring tenants).
# usage: tenant-cell.sh <off|on> <lockfd>
set -uo pipefail
arm=$1; fd=$2
R=/root/spill-receipts/c-day15
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
env_extra=(MEMRA_GPU_LOCK=/tmp/memra-gpu.lock CUDA_VISIBLE_DEVICES=0)
[ "$arm" = on ] && env_extra+=(MEMRA_KV_HOST_CONTRACTS=1)
sha256sum $R/bins/memra-server
env "${env_extra[@]}" bash tools/kv-host-tenant-reclaim-gate.sh --external-lock "$fd" fix /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf $R/bins/memra-server $R/tenant-$arm/ev
