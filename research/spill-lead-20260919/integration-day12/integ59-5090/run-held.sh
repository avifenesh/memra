#!/usr/bin/env bash
# integ59 (A days 37 to 40: finding 5 fixed, hash 1 off the owner as G4, the threaded fill T, item 5's Sources fault cells; S reverted; on main 9eb1326e9) on the
# local RTX 5090 Laptop GPU: integ54's cells unchanged (the identity, failure and fault gates carry M' and its bind re-hash); serve-smoke, the engine d2d_*, d2h_span* and h2d_span* native cells, the worker option_b_* and option_c_*
# native cells (serial: A's day-32 finding 3), then the five gates (the fault gate now carries day 32's
# promote-span-refusal cell), all under ONE collector hold of /tmp/memra-5090.lock (75 x `flock -w 120`: sized to outlast lane B day 34's first order, then take the card between its orders), then an idle
# check under the hold (no memra-server compute app, >= 20000 MiB free, 15 x 60 s; a foreign process is recorded, not
# waited out: every cell here is a correctness verdict). Either bound running out records every cell NOT RUN. Gates take
# the held FD via `--external-lock 9`. The binary is hashed again after serve-smoke's unconditional build. No gate, cell
# or threshold changed from the lanes' scripts. Executed-not-qualified.
# usage: run-held.sh <out_root> <model.gguf> <bin>
set -uo pipefail
ROOT=$1; MODEL=$2; BIN=$3
HERE=$(cd "$(dirname "$0")/../../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
TREE=$(git rev-parse HEAD)
sha256sum "$BIN" > "$ROOT/binary.sha256"
CELLS="serve-smoke engine-gpu-cells worker-span-cells identity-default-on fault-default fault-plain hit-off hit-on admit-mem-burst spec-ctx-edge"
not_run() {
    for c in $CELLS; do
        mkdir -p "$ROOT/$c"
        echo "$(date -u +%FT%TZ) $c NOT RUN: $1" | tee -a "$ROOT/battery.log" "$ROOT/$c/NOT-RUN"
    done
    exit 0
}
exec 9>"$MEMRA_GPU_LOCK"
held=0
for attempt in $(seq 1 75); do
    if flock -w 120 9; then held=1; break; fi
    echo "$(date -u +%FT%TZ) hold attempt $attempt: the lock stayed busy for 120 s" | tee -a "$ROOT/battery.log"
done
[ "$held" = 1 ] || not_run "the lock never came free in 75 bounded waits"
echo "$(date -u +%FT%TZ) collector hold taken on $MEMRA_GPU_LOCK (fd 9) tree=$TREE" | tee -a "$ROOT/battery.log"
idle=0
for attempt in $(seq 1 15); do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if ! grep -q 'memra-server' <<<"$apps" && [ "${free_mib:-0}" -ge 20000 ]; then
        [ -n "$apps" ] && echo "$(date -u +%FT%TZ) foreign co-tenant recorded: [${apps//$'\n'/; }]" | tee -a "$ROOT/battery.log"
        idle=1; break
    fi
    echo "$(date -u +%FT%TZ) idle wait $attempt under the hold: free=${free_mib}MiB apps=[${apps//$'\n'/; }]" | tee -a "$ROOT/battery.log"
    sleep 60
done
[ "$idle" = 1 ] || not_run "the card never went idle under the hold in 15 bounded waits"
run() { # $1 name $2 env-string $3.. command; the gates get $OUT/ev appended when $APPEND_EV=1
    local name=$1 envs=$2; shift 2
    local OUT=$ROOT/$name; mkdir -p "$OUT"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.before.csv" 2>&1
    echo "$(date -u +%FT%TZ) start $name env=[$envs]" | tee -a "$ROOT/battery.log"
    if [ "${APPEND_EV:-0}" = 1 ]; then
        # shellcheck disable=SC2086
        env $envs "$@" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$?
    else
        # shellcheck disable=SC2086
        env $envs "$@" > "$OUT/gate.log" 2>&1; rc=$?
    fi
    echo "$rc" > "$OUT/gate.exit"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv > "$OUT/card.after.csv" 2>&1
    {
        echo "cell=$name cmd=$*"; echo "env=$envs"; echo "lock=$MEMRA_GPU_LOCK owner=collector-fd9"
        echo "tree=$TREE"; echo "binary_sha256=$(cut -d' ' -f1 "$ROOT/binary.sha256")"; echo "model=$(basename "$MODEL")"
        echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"; echo "status=executed-not-qualified"
    } > "$OUT/CELL.txt"
    echo "$(date -u +%FT%TZ) done $name rc=$rc $(grep -hE 'GATE: |serve-smoke: |test result: ' "$OUT/gate.log" | tail -1)" | tee -a "$ROOT/battery.log"
}
run serve-smoke "" bash tools/serve-smoke.sh
# serve-smoke builds memra-server unconditionally (its own rule), so the binary every later cell
# runs is the one it just built: hash it again here (the integ49 first run named the pre-smoke
# binary in binary.sha256 while its later cells ran the rebuilt one).
sha256sum "$BIN" > "$ROOT/binary.sha256"
echo "$(date -u +%FT%TZ) binary after serve-smoke's build: $(cut -c1-16 "$ROOT/binary.sha256")" | tee -a "$ROOT/battery.log"
run engine-gpu-cells "" cargo test -p memra-engine --lib --offline -- --ignored --test-threads=1 d2d_ d2h_span h2d_
run worker-span-cells "" cargo test -p memra-server --lib --offline -- --ignored --test-threads=1 option_b_ option_c_
APPEND_EV=1 run identity-default-on "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_KV_HOST_CONTRACTS=1" bash tools/kv-host-spill-identity-gate.sh --external-lock 9 "$MODEL" "$BIN"
APPEND_EV=1 run fault-default "MEMRA_HOSTGATE_CACHE_MB=64" bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BIN"
APPEND_EV=1 run fault-plain "MEMRA_HOSTGATE_CACHE_MB=64 MEMRA_SERVE_SPEC=0" bash tools/kv-host-contract-fault-gate.sh --external-lock 9 "$MODEL" "$BIN"
APPEND_EV=1 run hit-off "" bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$BIN"
APPEND_EV=1 run hit-on "MEMRA_KV_HOST_CONTRACTS=1" bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$BIN"
APPEND_EV=1 run admit-mem-burst "" bash tools/admit-mem-burst-gate.sh "$MODEL" "$BIN"
APPEND_EV=1 run spec-ctx-edge "" bash tools/spec-ctx-edge-gate.sh "$MODEL" "$BIN"
flock -u 9
echo "$(date -u +%FT%TZ) collector hold released; INTEG59-5090-DONE" | tee -a "$ROOT/battery.log"
