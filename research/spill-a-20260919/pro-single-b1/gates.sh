#!/usr/bin/env bash
# B1's gate cell (DAY59 section 7 (a3)) on the b1 binary, door OFF and ON: the identity gate default and plain.
# lock hold: `tools/tier-battery.py --rig pro-single --external-lock --execute bash gates.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-b1
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/gates"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/gates/LOCK.json"
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=$R/bins/b1/memra-server
sha256sum "$BIN" | tee "$R/gates/binary.sha256"
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
run() { local name=$1; shift; "$@" > "$R/gates/$name.log" 2>&1; local rc=$?; echo "$rc" > "$R/gates/$name.exit"; echo "$(date -u +%FT%TZ) $name rc=$rc"; }
C=MEMRA_HOSTGATE_CACHE_MB=256
run identity-default-off env $C tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/identity-default-off"
run identity-default-on  env $C MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/identity-default-on"
run identity-plain-off   env $C MEMRA_SERVE_SPEC=0 tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/identity-plain-off"
run identity-plain-on    env $C MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/identity-plain-on"
echo gates-done
