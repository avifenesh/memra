#!/usr/bin/env bash
# Derived for the local RTX 5090 by rtx5090-derive.py (DAY68 section 1) from pro-single-l2/gates.sh at 21984b527: exact replacements only.
# Design L's gate cell (DAY63 section 1 (a)) on the l binary (one RTX PRO 6000 Blackwell), under the collector's
# hold: `tools/tier-battery.py --rig pro-single --external-lock --execute bash gates.sh @COLLECTOR_LOCK_FD@`. The
# identity gate default and plain door OFF and ON, the fault and contract fault gates default and plain, the pause gate
# and the hit gate door OFF and ON (all through the collector's FD).
set -uo pipefail
fd=$1
R=${A_OUT:?}
export PATH=/usr/local/cuda/bin:$PATH
cd "${A_TREE:?}" || exit 1
mkdir -p "$R/gates"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$R/gates/LOCK.json"
MODEL=${MEMRA_DAY38_MODEL:?}
BIN=$R/bins/l/memra-server
sha256sum "$BIN" | tee "$R/gates/binary.sha256"
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
run() { local name=$1; shift; "$@" > "$R/gates/$name.log" 2>&1; local rc=$?; echo "$rc" > "$R/gates/$name.exit"; echo "$(date -u +%FT%TZ) $name rc=$rc"; }
C=MEMRA_HOSTGATE_CACHE_MB=256
run identity-default-off env $C tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/identity-default-off"
run identity-default-on  env $C MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/identity-default-on"
run identity-plain-off   env $C MEMRA_SERVE_SPEC=0 tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/identity-plain-off"
run identity-plain-on    env $C MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/identity-plain-on"
run failure-off          env $C tools/kv-host-spill-failure-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/failure-off"
run failure-on           env $C MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-failure-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/failure-on"
run contract-fault       env $C tools/kv-host-contract-fault-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/contract-fault"
run contract-fault-plain env $C MEMRA_SERVE_SPEC=0 tools/kv-host-contract-fault-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/contract-fault-plain"
run pause-demote         env $C tools/kv-host-pause-demote-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/pause-demote"
run hitgate-off          tools/spec-on-cache-hit-gate.sh --external-lock "$fd" qwen "$MODEL" "$BIN" "$R/gates/hitgate-off"
run hitgate-on           env MEMRA_KV_HOST_CONTRACTS=1 tools/spec-on-cache-hit-gate.sh --external-lock "$fd" qwen "$MODEL" "$BIN" "$R/gates/hitgate-on"
echo gates-done
