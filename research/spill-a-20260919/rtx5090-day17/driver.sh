#!/usr/bin/env bash
# WP-A day 17 local RTX 5090 gate driver (DAY17.md pre-registration). Every door gate takes the
# canonical lock itself with `flock -n`, so a busy lock is a REFUSED exit 2 that this driver retries
# boundedly (15 x 120 s), never signalling the holder. Door OFF = env unset; door ON = MEMRA_KV_HOST_CONTRACTS=1.
# usage: driver.sh <model.gguf> <server_bin> <evidence_dir>
set -uo pipefail
MODEL=$1; BIN=$2; EV=$3
cd "$(dirname "$0")/../../.."
mkdir -p "$EV"
sha256sum "$BIN" | tee "$EV/binary.sha256"
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
run() { # name, then the command (env assignments allowed as leading words via env)
    local name=$1; shift
    local try rc
    for try in $(seq 1 15); do
        "$@" > "$EV/$name.log" 2>&1
        rc=$?
        if [[ $rc -eq 2 ]] && grep -q 'REFUSED: canonical GPU lock busy\|REFUSED' "$EV/$name.log"; then
            echo "$(date -u +%FT%TZ) $name lock busy, retry $try/15 in 120 s"; sleep 120; continue
        fi
        break
    done
    echo "$(date -u +%FT%TZ) $name rc=$rc"; echo "$rc" > "$EV/$name.exit"
}
C=MEMRA_HOSTGATE_CACHE_MB=64
run identity-default-off env $C tools/kv-host-spill-identity-gate.sh "$MODEL" "$BIN" "$EV/identity-default-off"
run identity-default-on  env $C MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh "$MODEL" "$BIN" "$EV/identity-default-on"
run identity-plain-off   env $C MEMRA_SERVE_SPEC=0 tools/kv-host-spill-identity-gate.sh "$MODEL" "$BIN" "$EV/identity-plain-off"
run identity-plain-on    env $C MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-identity-gate.sh "$MODEL" "$BIN" "$EV/identity-plain-on"
run failure-off          env $C tools/kv-host-spill-failure-gate.sh "$MODEL" "$BIN" "$EV/failure-off"
run failure-on           env $C MEMRA_KV_HOST_CONTRACTS=1 tools/kv-host-spill-failure-gate.sh "$MODEL" "$BIN" "$EV/failure-on"
run contract-fault       env $C tools/kv-host-contract-fault-gate.sh "$MODEL" "$BIN" "$EV/contract-fault"
run hitgate-off          tools/spec-on-cache-hit-gate.sh qwen "$MODEL" "$BIN" "$EV/hitgate-off"
run hitgate-on           env MEMRA_KV_HOST_CONTRACTS=1 tools/spec-on-cache-hit-gate.sh qwen "$MODEL" "$BIN" "$EV/hitgate-on"
run twin-off             python3 tools/prefix-newest-turn-fits-gate.py --model "$MODEL" --bin "$BIN" --out "$EV/twin-off"
run twin-on              env MEMRA_KV_HOST_CONTRACTS=1 python3 tools/prefix-newest-turn-fits-gate.py --model "$MODEL" --bin "$BIN" --out "$EV/twin-on"
# The door's GPU unit cells (option_b demote unwinds, option_c promote unwinds) on the copy-stream engine.
run gpu-unit-cells       flock -w 1800 /tmp/memra-5090.lock systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_
echo local-driver-done
