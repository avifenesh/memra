#!/usr/bin/env bash
# The design-V target-card gate cell (DAY47 section 1's (b), with the pause gate, plain and default boots: the day-36 set, the fault gate with every cell, source-flip,
# copy-phase-hit on the host-side hold and day 41's three sources-* cells, on the V tip binary) (one RTX PRO 6000 Blackwell), run under the collector's
# lock hold: `tools/tier-battery.py --rig pro-single --external-lock --execute bash gates.sh @COLLECTOR_LOCK_FD@`.
# Gates with an --external-lock arm inherit the FD; the hit gate has none and is run by hitgate.sh
# under flock on the canonical lock outside the collector (the day-16 shape).
set -uo pipefail
fd=$1
R=/root/spill-receipts/a-v
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-a
mkdir -p "$R/gates"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/gates/LOCK.json"
MODEL=${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=$R/bins/v/memra-server
sha256sum "$BIN" | tee "$R/gates/binary.sha256"
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
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
run twin-off             python3 tools/prefix-newest-turn-fits-gate.py --external-lock "$fd" --model "$MODEL" --bin "$BIN" --out "$R/gates/twin-off"
run twin-on              env MEMRA_KV_HOST_CONTRACTS=1 python3 tools/prefix-newest-turn-fits-gate.py --external-lock "$fd" --model "$MODEL" --bin "$BIN" --out "$R/gates/twin-on"
run pause-demote         env $C tools/kv-host-pause-demote-gate.sh --external-lock "$fd" "$MODEL" "$BIN" "$R/gates/pause-demote"
echo gates-done
