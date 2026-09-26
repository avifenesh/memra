#!/usr/bin/env bash
# Derived for the local RTX 5090 by rtx5090-derive.py (DAY68 section 1) from pro-single-l2/gates-red.sh at 21984b527: exact replacements only.
# The gate change's red arm (DAY63 section 4) under the collector's hold: the contract fault gate (the default arm) on the
# redgate binary (L' plus gate-red-arm.patch: a refused span drops its staging buffer). It must FAIL, with the two
# staging-fill checks among its failures.
# `tools/tier-battery.py --rig pro-single --external-lock --execute bash gates-red.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
fd=$1
R=${A_OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
cd "${A_TREE:?}" || exit 1
mkdir -p "$R/gates-red"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$R/gates-red/LOCK.json"
MODEL=${MEMRA_DAY38_MODEL:?}
BIN=$R/bins/redgate/memra-server
sha256sum "$BIN" | tee "$R/gates-red/binary.sha256"
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
env MEMRA_HOSTGATE_CACHE_MB=256 tools/kv-host-contract-fault-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates-red/contract-fault" > "$R/gates-red/contract-fault.log" 2>&1
echo "$?" > "$R/gates-red/contract-fault.exit"
echo "$(date -u +%FT%TZ) gates-red contract-fault rc=$(cat "$R/gates-red/contract-fault.exit")"
exit 0
