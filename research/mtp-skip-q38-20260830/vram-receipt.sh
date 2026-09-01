#!/usr/bin/env bash
# VRAM receipt for MEMRA_MTP_SKIP: boot the production-shape q38 server (dspark + trim) with
# the flag OFF and ON, record the server process's own nvidia-smi used_memory after load,
# plus the boot receipt lines. Exactness rig: numbers are per-process residency, not timing.
# usage: vram-receipt.sh <memra-server-bin> <q38.gguf> <dflash2_dir> <ranks.txt> <evidence_dir>
set -uo pipefail
BIN=$1; MODEL=$2; DRAFT=$3; RANKS=$4; EV=$5
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18135}
mkdir -p "$EV"
FAIL=0
SERVER_PID=""

stop() {
    [ -n "$SERVER_PID" ] || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 3
}
trap stop EXIT

arm() { # $1 name  $2 extra-env ("" for none)
    local name=$1 extra=$2
    local log="$EV/vram-$name.log"
    # shellcheck disable=SC2086
    setsid flock -w 900 "$LOCK" env CUDA_VISIBLE_DEVICES=0 \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
        MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        MEMRA_DSPARK_SPEC=1 "MEMRA_DSPARK_DRAFT=$DRAFT" "MEMRA_FRSPEC_TRIM=$RANKS" \
        $extra "$BIN" >"$log" 2>&1 &
    SERVER_PID=$!
    local up=0
    for _ in $(seq 1 300); do
        curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 2
    done
    if [ "$up" != 1 ]; then
        echo "FAIL[$name]: server did not come up"; tail -5 "$log" | sed 's/^/    /'; FAIL=1
        stop; return
    fi
    sleep 5 # let residency settle post-ready
    local spid mem
    spid=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    mem=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
        | awk -F', ' -v p="$spid" '$1==p {print $2}')
    if [ -z "$spid" ] || [ -z "$mem" ]; then
        echo "FAIL[$name]: could not attribute VRAM (pid=$spid mem=$mem)"; FAIL=1
    else
        echo "$name: server pid=$spid used_memory=${mem} MiB"
        echo "$mem" > "$EV/vram-$name.mib"
    fi
    grep -E "^\[mtp-skip\]|^\[frspec-trim\]|^\[dspark\]|^\[mtp-chain\]|^\[mtp-draft\]" "$log" \
        > "$EV/receipts-$name.txt" || true
    sed 's/^/    /' "$EV/receipts-$name.txt"
    stop
}

echo "== arm OFF (production shape, flag unset) =="
arm off ""
echo "== arm ON (production shape + MEMRA_MTP_SKIP=1) =="
arm on "MEMRA_MTP_SKIP=1"

if [ -f "$EV/vram-off.mib" ] && [ -f "$EV/vram-on.mib" ]; then
    OFF=$(cat "$EV/vram-off.mib"); ON=$(cat "$EV/vram-on.mib")
    echo "VRAM delta: off=${OFF} MiB on=${ON} MiB reclaimed=$((OFF - ON)) MiB"
fi
if [ "$FAIL" = 0 ]; then echo "vram-receipt: DONE"; else echo "vram-receipt: FAILED"; exit 1; fi
