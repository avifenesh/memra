#!/usr/bin/env bash
# WP-A day 25, Task 2: the hit gate on BOX3 under flock on the canonical lock, door ON only (the OFF arm's
# evidence is day 24's; the typed settle line never prints under OFF). The ON arm arms the host tier itself
# (C day 27: MEMRA_KV_HOST_MB=8192 exported by the gate under MEMRA_KV_HOST_CONTRACTS=1). Bounded lock retries,
# the holder never signalled.
set -uo pipefail
R=/root/spill-receipts/a-day25
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a || exit 1
mkdir -p "$R/gates"
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/memra-server
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
sha256sum "$BIN" | tee "$R/gates/binary.sha256"
arm=on
extra="MEMRA_KV_HOST_CONTRACTS=1"
for try in $(seq 1 15); do
  env $extra tools/spec-on-cache-hit-gate.sh qwen "$MODEL" "$BIN" "$R/gates/hitgate-$arm" > "$R/gates/hitgate-$arm.log" 2>&1; rc=$?
  if [ $rc -eq 2 ] && grep -q REFUSED "$R/gates/hitgate-$arm.log"; then echo "$(date -u +%FT%TZ) hitgate-$arm lock busy, retry $try/15"; sleep 120; continue; fi
  break
done
echo "$rc" > "$R/gates/hitgate-$arm.exit"; echo "$(date -u +%FT%TZ) hitgate-$arm rc=$rc"
echo hitgate-done
