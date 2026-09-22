#!/usr/bin/env bash
# WP-A day 25, Task 1: the double park (C day 29's finding; memra#536 Move 2). Pre-registered in DAY25.md
# before this ran. The day-16 stall script's PROMOTE arm, door ON against door OFF, ONE collector lock hold on
# the target card. The two arms cannot share a boot (the door is a boot flag), so the A/B interleaves across
# BOOTS: order 1 = ON OFF ON OFF ON OFF ON OFF ON OFF, order 2 = OFF ON OFF ON OFF ON OFF ON OFF ON (five boots per
# arm per order, twenty boots). Every boot is the day-16 script's `off` boot (MEMRA_PREFIX_CACHE_MB=256
# MEMRA_KV_HOST_MB=8192 MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4) plus MEMRA_KV_HOST_CONTRACTS=1 on
# the ON boots, running lane A's harness stall_cell.py byte-for-byte, --mode promote --n 5 (its own idle/arm
# interleave in both orders inside the boot). Every receipt replayed (`STALL REPLAY`). Every cell is
# executed-not-qualified development evidence. No host, id or price here.
# usage: double-park.sh <lockfd> <tree> <receipts_root> <model.gguf> <bin>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN=$5
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$TREE" || exit 1
EV=$R/double-park/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum "$BIN" | tee "$EV/binary.sha256"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "harness_sha256=$(sha256sum research/spill-a-20260919/stall_cell.py | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "boots=o1:ON,OFF,ON,OFF,ON,OFF,ON,OFF,ON,OFF o2:OFF,ON,OFF,ON,OFF,ON,OFF,ON,OFF,ON"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.before.csv" 2>&1
PORT=${MEMRA_GATE_PORT:-18132}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 extra-env-string  $2 log
    memra_port_guard double-park "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving, refusing to boot over it"; return 1
    fi
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 \
        $1 "$BIN" >"$2" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died during boot:"; tail -20 "$2"; return 1; }
        sleep 2
    done
    echo "server never became ready"; return 1
}
stop() {
    [[ -n $SERVER_PID ]] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop EXIT
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
card() { nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used --format=csv > "$1" 2>&1; }
H=research/spill-a-20260919/stall_cell.py
BASE="MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192"
: > "$EV/marks.tsv"
: > "$EV/replays.log"
rc=0
# one boot: $1 dir  $2 arm (on|off)
one_boot() {
    local D=$1 arm=$2 extra=$BASE
    [ "$arm" = on ] && extra="$BASE MEMRA_KV_HOST_CONTRACTS=1"
    mkdir -p "$D"
    echo "arm=$arm env=$extra" > "$D/BOOT.txt"
    card "$D/card.before.csv"
    mark "boot-$(basename "$D")"
    if ! boot "$extra" "$D/server.log"; then
        echo "boot $(basename "$D") arm=$arm FAILED" | tee -a "$EV/replays.log"; echo "boot-failed" >> "$D/BOOT.txt"; return 1
    fi
    mark "ready-$(basename "$D")"
    python3 $H --port "$PORT" --mode promote --n 5 --server-log "$D/server.log" --out "$D/promote" --tag "stall-promote-$arm" | tee "$D/promote.log" || rc=1
    mark "promote-done-$(basename "$D")"
    stop; mark "stopped-$(basename "$D")"
    card "$D/card.after.csv"
    { echo "== $(basename "$D")/promote arm=$arm"; python3 $H --replay "$D/promote/receipt.json"; } | tee -a "$EV/replays.log" || rc=1
    return 0
}
i=0
for arm in on off on off on off on off on off; do
    i=$((i+1)); D=$EV/o1/$(printf 'b%02d-%s' "$i" "$arm")
    one_boot "$D" "$arm" || rc=1
done
for arm in off on off on off on off on off on; do
    i=$((i+1)); D=$EV/o2/$(printf 'b%02d-%s' "$i" "$arm")
    one_boot "$D" "$arm" || rc=1
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
echo "double-park rc=$rc" | tee "$EV/exit.txt"
exit $rc
