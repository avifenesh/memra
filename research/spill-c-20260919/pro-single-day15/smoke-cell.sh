#!/usr/bin/env bash
# serve-smoke (plain + cache-metering arms) on the day-15 binary; arm off|on sets the door.
set -uo pipefail
arm=$1
R=/root/spill-receipts/c-day15
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
env_extra=()
[ "$arm" = on ] && env_extra=(MEMRA_KV_HOST_CONTRACTS=1)
sha256sum target/release/memra-server
env CUDA_VISIBLE_DEVICES=0 "${env_extra[@]}" bash tools/serve-smoke.sh /root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf /nonexistent-draft
rc=$?
cp /tmp/serve-smoke.log "$R/smoke-$arm-server.log" 2>/dev/null; rm -f /tmp/serve-smoke.log
exit $rc
