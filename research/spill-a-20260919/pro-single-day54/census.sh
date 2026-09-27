#!/usr/bin/env bash
# DAY54's census (DAY54.md section 1): the door-ON gates on the tip binary, their server logs kept for the on-tick
# publish lines, in ONE collector hold: `tier-battery.py --rig pro-single --external-lock --execute bash census.sh @COLLECTOR_LOCK_FD@`.
# The identity gate door ON in the default and plain boots, and the pause gate; the hit gate (no --external-lock arm)
# runs from hitgate.sh under its own flock. Executed-not-qualified.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-d54
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/census"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/census/LOCK.json"
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=$R/bins/tip/memra-server
sha256sum "$BIN" | tee "$R/census/binary.sha256"
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
run() { local name=$1; shift; "$@" > "$R/census/$name.log" 2>&1; local rc=$?; echo "$rc" > "$R/census/$name.exit"; echo "$(date -u +%FT%TZ) $name rc=$rc"; }
C=MEMRA_HOSTGATE_CACHE_MB=256
run identity-default-on env $C MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/census/identity-default-on"
run identity-plain-on   env $C MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/census/identity-plain-on"
run pause-demote        env $C tools/kv-host-pause-demote-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/census/pause-demote"
echo census-done
