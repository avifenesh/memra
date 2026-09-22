#!/usr/bin/env bash
# Day 23 promote stall cell on main's tree (owed to the HOSTPREFIX door review by integ32: A's day-18
# promote receipt was taken with the owner-thread spin; #627's parked-only wait changes timing, not bytes).
# Lane A's day-16/day-18 shape, unchanged: `stall_cell.py --mode demote` then `--mode promote` per boot,
# N=5 per arm per order, both orders, MEMRA_SERVE_SPEC=0 so the tenant emits one token per tick. Two boots
# (door OFF, then door ON) inside ONE collector lock hold, so OFF against ON is a same-window pair on one
# tree; A's script held the lock once per boot. Harness: research/spill-a-20260919/stall_cell.py as merged
# on this tree (byte-identical to lane A's tip). Path changes from A's pro-single-day18/stall-cell.sh: the
# worktree, the receipts root and the binary are arguments; the tree SHA and compute apps are recorded.
# Every cell is executed-not-qualified development evidence. No host, id or price here.
# usage: day23-stall-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <server_bin> off|on|both
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN=$5; which=$6
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$TREE" || exit 1
EV=$R/stall/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum "$BIN" | tee "$EV/binary.sha256"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "harness_sha256=$(sha256sum research/spill-a-20260919/stall_cell.py | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "boots=$which"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.before.csv" 2>&1
PORT=${MEMRA_GATE_PORT:-18132}
. tools/port-guard.sh
SERVER_PID=""
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
H=research/spill-a-20260919/stall_cell.py
: > "$EV/marks.tsv"
: > "$EV/replays.log"
rc=0
case $which in off) BOOTS=(off) ;; on) BOOTS=(on) ;; both) BOOTS=(off on) ;;
  *) echo "usage: day23-stall-cell.sh <lockfd> <tree> <receipts_root> <model> <bin> off|on|both"; exit 2 ;; esac
for b in "${BOOTS[@]}"; do
    D=$EV/$b; mkdir -p "$D"
    extra="MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192"
    [ "$b" = on ] && extra="$extra MEMRA_KV_HOST_CONTRACTS=1"
    mark "boot-$b"; boot "$extra" "$D/server.log" || { rc=1; echo "boot $b failed" | tee -a "$EV/replays.log"; break; }; mark "ready-$b"
    python3 $H --port "$PORT" --mode demote --server-log "$D/server.log" --out "$D/demote" --tag "stall-demote-$b" | tee "$D/demote.log" || rc=1
    mark "demote-done-$b"
    python3 $H --port "$PORT" --mode promote --server-log "$D/server.log" --out "$D/promote" --tag "stall-promote-$b" | tee "$D/promote.log" || rc=1
    mark "promote-done-$b"
    stop; mark "stopped-$b"
    for arm in demote promote; do
        { echo "== $b/$arm"; python3 $H --replay "$D/$arm/receipt.json"; } | tee -a "$EV/replays.log" || rc=1
    done
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
echo "stall-$which rc=$rc" | tee "$EV/exit.txt"
exit $rc
