#!/usr/bin/env bash
# Day 29 decision cell (i) of Move 1 (memra#536, OWNER-THREAD-OFFLOAD.md "What Move 1 still owes" item 4;
# pre-registered in DAY29.md before this ran). Same window, both classes (demote D2H, promote H2D), door ON in
# both arms, ONE collector lock hold:
#   arm X  today's tree, the copy-stream program (bin_x)
#   arm Y  the day-16 tree 1646d421b built in its own worktree, the owner-stream program (bin_y)
# Every boot is lane A's day-16 script's `on` boot (MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192
# MEMRA_KV_HOST_CONTRACTS=1 MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4) running A's harness
# stall_cell.py byte-for-byte, --mode demote then --mode promote, N=5 per arm per order inside the boot.
# The A/B interleave is across boots: order 1 = X Y X Y X Y X Y X Y, order 2 = Y X Y X Y X Y X Y X (five boots
# per arm per order, twenty boots). One DRY boot of arm Y first (boot, /v1/models, --mode demote --n 1, stop,
# replay), recorded under ev/dry-y and part of no quantity. Every receipt replayed (`STALL REPLAY`).
# Every cell is executed-not-qualified development evidence. No host, id or price here.
# usage: day29-stall-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <bin_x> <bin_y> <tree_y>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN_X=$5; BIN_Y=$6; TREE_Y=$7
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$TREE" || exit 1
EV=$R/stall/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum "$BIN_X" | tee "$EV/binary-x.sha256"
sha256sum "$BIN_Y" | tee "$EV/binary-y.sha256"
{
    echo "tree_x=$(git rev-parse HEAD)"
    echo "tree_y=$(git -C "$TREE_Y" rev-parse HEAD)"
    echo "harness_sha256=$(sha256sum research/spill-a-20260919/stall_cell.py | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "boots=dry-y o1:XYXYXYXYXY o2:YXYXYXYXYX"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
{
    echo "git diff --stat 1646d421b 653c997f4 -- crates/ Cargo.toml Cargo.lock (empty = equal):"
    git diff --stat 1646d421b 653c997f4 -- crates/ Cargo.toml Cargo.lock
    echo "lines=$(git diff --stat 1646d421b 653c997f4 -- crates/ Cargo.toml Cargo.lock | wc -l)"
} > "$EV/day16-crates-equality.txt"
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
# one boot: $1 dir  $2 arm letter (x|y)  $3 harness N  $4 modes (space separated)
one_boot() {
    local D=$1 arm=$2 n=$3 modes=$4 m
    mkdir -p "$D"
    if [ "$arm" = x ]; then BIN=$BIN_X; else BIN=$BIN_Y; fi
    echo "arm=$arm bin=$BIN n=$n modes=$modes" > "$D/BOOT.txt"
    card "$D/card.before.csv"
    mark "boot-$(basename "$D")"
    if ! boot "$EXTRA" "$D/server.log"; then
        echo "boot $(basename "$D") arm=$arm FAILED" | tee -a "$EV/replays.log"; echo "boot-failed" >> "$D/BOOT.txt"; return 1
    fi
    mark "ready-$(basename "$D")"
    for m in $modes; do
        python3 $H --port "$PORT" --mode "$m" --n "$n" --server-log "$D/server.log" --out "$D/$m" --tag "stall-$m-$arm" | tee "$D/$m.log" || rc=1
        mark "$m-done-$(basename "$D")"
    done
    stop; mark "stopped-$(basename "$D")"
    card "$D/card.after.csv"
    for m in $modes; do
        { echo "== $(basename "$D")/$m arm=$arm"; python3 $H --replay "$D/$m/receipt.json"; } | tee -a "$EV/replays.log" || rc=1
    done
    return 0
}
# the dry boot of arm Y: proves the day-16 tree boots and answers under today's artifact and CUDA
if ! one_boot "$EV/dry-y" y 1 "demote"; then
    echo "DRY BOOT OF ARM Y FAILED: the day-16 tree does not boot; the cell stops (report, no patch)" | tee "$EV/DRY-FAILED"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
    echo "stall-day29 rc=2 dry-failed" | tee "$EV/exit.txt"; exit 2
fi
i=0
for arm in x y x y x y x y x y; do
    i=$((i+1)); D=$EV/o1/$(printf 'b%02d-%s' "$i" "$arm")
    one_boot "$D" "$arm" 5 "demote promote" || rc=1
done
for arm in y x y x y x y x y x; do
    i=$((i+1)); D=$EV/o2/$(printf 'b%02d-%s' "$i" "$arm")
    one_boot "$D" "$arm" 5 "demote promote" || rc=1
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
echo "stall-day29 rc=$rc" | tee "$EV/exit.txt"
exit $rc
