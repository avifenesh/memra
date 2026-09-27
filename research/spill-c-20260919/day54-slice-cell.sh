#!/usr/bin/env bash
# Day 54 cell `slices` (OWED C4, DAY54.md, pre-registered before any slice binary was built or booted): which
# first-parent slice between 0713c1a79 and the integ38 tip moved the promote intruder's inline demote landing (the
# day-29 double park). Target card, ONE collector lock hold, eight memra-server binaries (e0 = 0713c1a79, s1 to s7
# the seven crate-changing slices in first-parent order), every boot lane A's day-16 script's `on` boot
# (MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1 MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192
# MEMRA_MAX_SESSIONS=4) running A's harness stall_cell.py byte-for-byte, --mode promote only, N=5 per arm per order
# inside the boot. Interleave across boots: order 1 = e0 s1 ... s7 five times, order 2 = s7 ... e0 five times (ten
# boots per binary). One DRY boot of e0 first (the oldest tree: boot, /v1/models, --mode promote --n 1, stop,
# replay), part of no quantity. Every receipt replayed (`STALL REPLAY`). Executed-not-qualified development
# evidence. No host, id or price here.
# usage: day54-slice-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <bins_dir>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BINS=$5
LABELS=(e0 s1 s2 s3 s4 s5 s6 s7)
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$TREE" || exit 1
EV=$R/slices/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
for l in "${LABELS[@]}"; do
    [ -x "$BINS/memra-server-$l" ] || { echo "missing $BINS/memra-server-$l"; exit 2; }
    sha256sum "$BINS/memra-server-$l"
done | tee "$EV/binary.sha256"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "harness_sha256=$(sha256sum research/spill-a-20260919/stall_cell.py | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "labels=${LABELS[*]} boots=dry-e0 o1:(e0..s7)x5 o2:(s7..e0)x5 mode=promote"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.before.csv" 2>&1
PORT=${MEMRA_GATE_PORT:-18132}
. tools/port-guard.sh
SERVER_PID=""
BIN=""
boot() { # $1 extra-env-string  $2 log
    memra_port_guard stall-cell "$PORT" MEMRA_GATE_PORT || return 1
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
EXTRA="MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1"
: > "$EV/marks.tsv"
: > "$EV/replays.log"
rc=0
# one boot: $1 dir  $2 label  $3 harness N  $4 modes (space separated)
one_boot() {
    local D=$1 label=$2 n=$3 modes=$4 m
    mkdir -p "$D"
    BIN=$BINS/memra-server-$label
    echo "label=$label bin=$BIN n=$n modes=$modes" > "$D/BOOT.txt"
    card "$D/card.before.csv"
    mark "boot-$(basename "$D")"
    if ! boot "$EXTRA" "$D/server.log"; then
        echo "boot $(basename "$D") label=$label FAILED" | tee -a "$EV/replays.log"; echo "boot-failed" >> "$D/BOOT.txt"; return 1
    fi
    mark "ready-$(basename "$D")"
    for m in $modes; do
        python3 $H --port "$PORT" --mode "$m" --n "$n" --server-log "$D/server.log" --out "$D/$m" --tag "stall-$m-$label" | tee "$D/$m.log" || rc=1
        mark "$m-done-$(basename "$D")"
    done
    stop; mark "stopped-$(basename "$D")"
    card "$D/card.after.csv"
    for m in $modes; do
        { echo "== $(basename "$D")/$m label=$label"; python3 $H --replay "$D/$m/receipt.json"; } | tee -a "$EV/replays.log" || rc=1
    done
    return 0
}
if ! one_boot "$EV/dry-e0" e0 1 "promote"; then
    echo "DRY BOOT OF e0 FAILED: the oldest tree does not boot; the cell stops (report, no patch)" | tee "$EV/DRY-FAILED"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
    echo "slices-day54 rc=2 dry-failed" | tee "$EV/exit.txt"; exit 2
fi
i=0
for _rep in 1 2 3 4 5; do
    for l in "${LABELS[@]}"; do
        i=$((i+1)); one_boot "$EV/o1/$(printf 'b%02d-%s' "$i" "$l")" "$l" 5 "promote" || rc=1
    done
done
for _rep in 1 2 3 4 5; do
    for ((k = ${#LABELS[@]} - 1; k >= 0; k--)); do
        l=${LABELS[$k]}; i=$((i+1)); one_boot "$EV/o2/$(printf 'b%02d-%s' "$i" "$l")" "$l" 5 "promote" || rc=1
    done
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
echo "slices-day54 rc=$rc" | tee "$EV/exit.txt"
exit $rc
