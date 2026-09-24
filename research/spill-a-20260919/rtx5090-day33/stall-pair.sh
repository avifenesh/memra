#!/usr/bin/env bash
# WP-A day 33, the 5090 development reading of DAY33 section 2 (not a clause): the stall harness's promote arm
# (stall_cell.py byte for byte, --mode promote --n 5) on the local RTX 5090 Laptop GPU with the 9B, the day-26
# double-park boot environment except the prefix budget (MEMRA_PREFIX_CACHE_MB=64 so the 54 MB entries evict, the
# gates' shape on this card). Five boots under ONE hold of /tmp/memra-5090.lock (bounded, 15 x flock -w 120; lane B
# day 33 shares the card): d32-on, d33-on, off, d33-on, d32-on (the day-32 H2D binary, the day-33 binary, the door OFF
# on the day-33 binary). Reads per boot: the promote's poll count and submission-to-completion, `promote_in`, the
# owner segment, the tenant stall and the intruder's e2e. No 5090 number is compared to the target card's.
# usage: stall-pair.sh <out_root> <model.gguf> <bin_d32> <bin_d33>
set -uo pipefail
ROOT=$1; MODEL=$2; B32=$3; B33=$4
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
mkdir -p "$ROOT"
sha256sum "$B32" "$B33" > "$ROOT/binaries.sha256"
exec 9>/tmp/memra-5090.lock
got=0
for try in $(seq 1 15); do
    if flock -w 120 9; then got=1; break; fi
    echo "$(date -u +%FT%TZ) lock busy, wait $try/15" | tee -a "$ROOT/stall.log"
done
[ $got = 1 ] || { echo "$(date -u +%FT%TZ) NOT RUN: the lock never freed" | tee -a "$ROOT/stall.log"; exit 2; }
echo "$(date -u +%FT%TZ) hold start; card: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')" | tee -a "$ROOT/stall.log"
PORT=${MEMRA_GATE_PORT:-18133}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 bin $2 extra-env $3 log
    memra_port_guard stall-pair "$PORT" MEMRA_GATE_PORT || return 1
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
        MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=64 MEMRA_KV_HOST_MB=8192 \
        $2 "$1" > "$3" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died during boot"; tail -20 "$3"; return 1; }
        sleep 2
    done
    return 1
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
i=0
for arm in d32-on d33-on off d33-on d32-on; do
    i=$((i+1)); D=$ROOT/$(printf 'b%02d-%s' "$i" "$arm"); mkdir -p "$D"
    case $arm in
        d32-on) bin=$B32; extra=MEMRA_KV_HOST_CONTRACTS=1 ;;
        d33-on) bin=$B33; extra=MEMRA_KV_HOST_CONTRACTS=1 ;;
        off) bin=$B33; extra=MEMRA_KV_HOST_CONTRACTS=0 ;;
    esac
    echo "arm=$arm bin=$(sha256sum "$bin" | cut -c1-16) env=$extra" > "$D/BOOT.txt"
    if ! boot "$bin" "$extra" "$D/server.log"; then echo "$(date -u +%FT%TZ) $arm boot FAILED" | tee -a "$ROOT/stall.log"; continue; fi
    python3 research/spill-a-20260919/stall_cell.py --port "$PORT" --mode promote --n 5 --server-log "$D/server.log" \
        --out "$D/promote" --tag "stall-promote-$arm" > "$D/promote.log" 2>&1
    echo "$(date -u +%FT%TZ) $arm rc=$? $(grep -h 'STALL rule' "$D/promote.log" | cut -c1-200)" | tee -a "$ROOT/stall.log"
    stop
    python3 research/spill-a-20260919/stall_cell.py --replay "$D/promote/receipt.json" >> "$ROOT/replays.log" 2>&1
done
echo "$(date -u +%FT%TZ) hold end; card: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')" | tee -a "$ROOT/stall.log"
echo "STALL-PAIR-DONE" >> "$ROOT/stall.log"
