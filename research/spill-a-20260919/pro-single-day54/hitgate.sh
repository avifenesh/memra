#!/usr/bin/env bash
# DAY54's census, the hit gate door ON (no --external-lock arm): under flock on the canonical lock. The ON arm arms the
# host tier itself (C day 27).
set -uo pipefail
R=/root/spill-receipts/a-d54
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/census"
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=$R/bins/tip/memra-server
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
for try in $(seq 1 15); do
  env MEMRA_KV_HOST_CONTRACTS=1 tools/spec-on-cache-hit-gate.sh qwen "$MODEL" "$BIN" "$R/census/hitgate-on" > "$R/census/hitgate-on.log" 2>&1; rc=$?
  if [ $rc -eq 2 ] && grep -q REFUSED "$R/census/hitgate-on.log"; then echo "$(date -u +%FT%TZ) hitgate-on lock busy, retry $try/15"; sleep 120; continue; fi
  break
done
echo "$rc" > "$R/census/hitgate-on.exit"; echo "$(date -u +%FT%TZ) hitgate-on rc=$rc"
