#!/usr/bin/env bash
# The hit gate on the G4 target card (DAY42 section 1's (b)), on the S2 tip binary (no --external-lock arm): under flock on the canonical lock, door OFF then ON. The ON arm
# arms the host tier itself (C day 27: MEMRA_KV_HOST_MB=8192 exported by the gate under MEMRA_KV_HOST_CONTRACTS=1).
set -uo pipefail
R=/root/spill-receipts/a-s2
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/gates"
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=$R/bins/s2/memra-server
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
