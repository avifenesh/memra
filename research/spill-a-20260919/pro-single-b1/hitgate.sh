#!/usr/bin/env bash
# B1's hit gate (DAY59 section 7 (a3)) on the b1 binary, under flock on the canonical lock, door OFF then ON.
set -uo pipefail
R=/root/spill-receipts/a-b1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/gates"
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=$R/bins/b1/memra-server
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
for arm in off on; do
  extra=""; [ $arm = on ] && extra="MEMRA_KV_HOST_CONTRACTS=1"
  for try in $(seq 1 15); do
    env $extra tools/spec-on-cache-hit-gate.sh qwen "$MODEL" "$BIN" "$R/gates/hitgate-$arm" > "$R/gates/hitgate-$arm.log" 2>&1; rc=$?
    if [ $rc -eq 2 ] && grep -q REFUSED "$R/gates/hitgate-$arm.log"; then echo "$(date -u +%FT%TZ) hitgate-$arm lock busy, retry $try/15"; sleep 120; continue; fi
    break
  done
  echo "$rc" > "$R/gates/hitgate-$arm.exit"; echo "$(date -u +%FT%TZ) hitgate-$arm rc=$rc"
done
echo hitgate-done
