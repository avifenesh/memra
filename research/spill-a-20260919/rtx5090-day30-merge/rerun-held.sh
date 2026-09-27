#!/usr/bin/env bash
# WP-A day 30, the merged tree (origin/main after #652 merged into the lane) on the local RTX 5090 Laptop GPU: the
# five cells of battery-5090.sh under ONE collector hold of /tmp/memra-5090.lock. Written after that script's first
# pass on this tree (battery.log): the identity and fault gates refused on the canonical lock (their `flock -n` met
# another lane's hold while the card idled between that lane's boots) and hit-off's launch wrapper timed out
# (`flock -w 300`, empty server log). Same gates, env, binary and artifact as battery-5090.sh; each gate takes the held
# FD through its own `--external-lock 9` argument. The hold is taken with a bounded wait (15 x `flock -w 120`); once
# held, the card must carry no compute app and at least 20000 MiB free (bounded, 15 x 60 s). Either bound running
# out records every cell NOT RUN. The holder of the lock is never inspected beyond nvidia-smi's own listing and never
# signalled. Executed-not-qualified. usage: rerun-held.sh <out_root> <model.gguf> <bin>
set -uo pipefail
ROOT=$1; MODEL=$2; BIN=$3
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
TREE=$(git rev-parse HEAD)
sha256sum "$BIN" > "$ROOT/binary.sha256"
CELLS="identity-default-on-held fault-default-held fault-plain-held hit-off-held hit-on-held"
not_run() {
    for c in $CELLS; do
        mkdir -p "$ROOT/$c"
        echo "$(date -u +%FT%TZ) $c NOT RUN: $1" | tee -a "$ROOT/battery.log" "$ROOT/$c/NOT-RUN"
    done
    exit 0
}
exec 9>"$MEMRA_GPU_LOCK"
held=0
for attempt in $(seq 1 15); do
    if flock -w 120 9; then held=1; break; fi
    echo "$(date -u +%FT%TZ) hold attempt $attempt: the lock stayed busy for 120 s" | tee -a "$ROOT/battery.log"
done
[ "$held" = 1 ] || not_run "the lock never came free in 15 bounded waits"
echo "$(date -u +%FT%TZ) collector hold taken on $MEMRA_GPU_LOCK (fd 9)" | tee -a "$ROOT/battery.log"
idle=0
for attempt in $(seq 1 15); do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$apps" ] && [ "${free_mib:-0}" -ge 20000 ]; then idle=1; break; fi
    echo "$(date -u +%FT%TZ) idle wait $attempt under the hold: free=${free_mib}MiB apps=[${apps//$'\n'/; }]" | tee -a "$ROOT/battery.log"
    sleep 60
done
[ "$idle" = 1 ] || not_run "the card never went idle under the hold in 15 bounded waits"
cell() { # $1 name $2 env-string $3.. command (the gate, with --external-lock 9; $OUT/ev is its evidence dir)
    local name=$1 envs=$2; shift 2
    local OUT=$ROOT/$name; mkdir -p "$OUT"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.before.csv" 2>&1
    echo "$(date -u +%FT%TZ) start $name tree=$TREE env=[$envs]" | tee -a "$ROOT/battery.log"
    # shellcheck disable=SC2086
    env $envs "$@" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$?
    echo "$rc" > "$OUT/gate.exit"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.after.csv" 2>&1
    {
        echo "cell=$name gate=$1"; echo "env=$envs"; echo "lock=$MEMRA_GPU_LOCK owner=collector-fd9"
        echo "tree=$TREE"; echo "binary_sha256=$(cut -d' ' -f1 "$ROOT/binary.sha256")"; echo "model=$(basename "$MODEL")"
        echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"; echo "status=executed-not-qualified"
    } > "$OUT/CELL.txt"
    echo "$(date -u +%FT%TZ) done $name rc=$rc $(grep -hE 'GATE: ' "$OUT/gate.log" | tail -1)" | tee -a "$ROOT/battery.log"
}
cell identity-default-on-held "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BIN"
cell fault-default-held "MEMRA_HOSTGATE_CACHE_MB=64" bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BIN"
cell fault-plain-held "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_SERVE_SPEC=0" bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BIN"
cell hit-off-held "" bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$BIN"
cell hit-on-held "MEMRA_KV_HOST_CONTRACTS=1" bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$BIN"
flock -u 9
echo "$(date -u +%FT%TZ) collector hold released; LOCAL-HELD-RERUN-DONE" | tee -a "$ROOT/battery.log"
