#!/usr/bin/env bash
# Day 31: the door's demote and promote PAIR on the local RTX 5090 Laptop GPU (HOSTPREFIX-DOOR.md section D
# item 5; the packet's "not run" row). The day-16 wc-pair cell (pro-single-day16/wc-cell.sh) adapted to this
# card and the 9B artifact: door OFF versus ON demote and promote wall times for the SAME two entries, N=5 per
# arm per order, BOTH orders (OFF,ON then ON,OFF), under ONE collector lock hold on /tmp/memra-5090.lock (the
# collector's 250 ms telemetry plus this script's own 1 s sampler). Four server boots. Per boot: r1 P_A seeds
# E_A; r2 P_B seeds E_B and evicts E_A (demote 1); r3..r7 alternate P_A/P_B: each is a host hit that promotes
# and whose insert evicts the other entry (a demote inside the promote window). Demotes r2..r6 (N=5) and
# promotes r3..r7 (N=5) are the observations; the server log is the receipt. MEMRA_KV_HOST_VERIFY unset (the
# production promote shape), MEMRA_SERVE_SPEC unset (the day-16 boot shape: the entries are spec-boundary,
# draft-bearing, `items=18` on the 9B). Budgets for this card and model: MEMRA_PREFIX_CACHE_MB=64 (one 53.8 MB
# entry fits, two do not; the gates' budget on this card since day 23), MEMRA_KV_HOST_MB=8192.
# Before the boots, one short receipt names the pinned destination arm this device resolves to:
# `tier-transfer-gate roundtrip` prints `PINNED-DEFAULT device=... kind=... flags=...` (run only under the
# collector, as its header requires; it is under the collector's hold here).
# Every cell is executed-not-qualified development evidence. No host, id or price here.
# usage: day31-wc-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <server_bin>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN=$5
cd "$TREE" || exit 1
EV=$R/pair/wc-pair/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-5090.lock --owner collector > "$EV/LOCK.json"
sha256sum "$BIN" | tee "$EV/binary.sha256"
CACHE_MB=64
HOST_MB=8192
PORT=${MEMRA_GATE_PORT:-18131}
{
    echo "tree=$(git rev-parse HEAD)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "cache_mb=$CACHE_MB host_mb=$HOST_MB port=$PORT"
    echo "orders=2 n_per_arm_per_order=5 boots=4 requests_per_boot=7"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
snap() { # label
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.$1.csv" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw,pstate --format=csv > "$EV/card.$1.csv" 2>&1
}
snap before
# The pinned destination arm this device resolves to, printed by the transfer gate's setup (PinnedKind::for_device).
GATE_BIN=$TREE/target/release/tier-transfer-gate
if [ -x "$GATE_BIN" ]; then
    CUDA_VISIBLE_DEVICES=0 "$GATE_BIN" roundtrip > "$EV/pinned-default.log" 2>&1
    echo "rc=$?" >> "$EV/pinned-default.log"
else
    echo "tier_transfer_gate binary absent at $GATE_BIN; the pinned arm is read from the device name and the unit table only" > "$EV/pinned-default.log"
fi
grep -h 'PINNED-DEFAULT' "$EV/pinned-default.log" || true
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 extra-env-string  $2 log
    memra_port_guard day31-wc-cell "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving, refusing to boot over it"; return 1
    fi
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        "MEMRA_PREFIX_CACHE_MB=$CACHE_MB" "MEMRA_KV_HOST_MB=$HOST_MB" $1 "$BIN" >"$2" 2>&1 &
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
SAMPLER=""
stop_sampler() {
    [[ -n $SAMPLER ]] || return 0
    kill -TERM "$SAMPLER" 2>/dev/null || true
    wait "$SAMPLER" 2>/dev/null || true
    SAMPLER=""
}
cleanup() { stop; stop_sampler; }
trap cleanup EXIT
P_A="You are indexing the survey logs of a coastal tide-gauge network. For each of the twelve \
stations, ordered north to south, report the gauge type, the datum epoch, the sampling \
interval in minutes, the last calibration date, the responsible technician role, and the \
anomaly that would force an out-of-cycle calibration. Be systematic and terse; do not skip \
a station. After the twelve stations, add a short paragraph on network-wide drift checks."
P_B="Draft the commissioning checklist for a small hydroelectric turbine hall. Cover, in \
order: penstock inspection, wicket-gate travel, governor response, generator insulation, \
thrust-bearing temperature rise, cooling-water flow, overspeed trip, and grid-synchronization \
tests. For each item name the instrument used, the acceptance threshold, the sign-off role, \
and the failure symptom that would halt commissioning. Be systematic and terse throughout."
req() { # $1 prompt $2 out-json
    python3 - "$PORT" "$1" "$2" <<'PYEOF'
import json, sys, urllib.request
port, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": "gate", "prompt": prompt, "max_tokens": 48, "temperature": 0}
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
json.dump(json.load(r), open(out, "w"), indent=1)
PYEOF
}
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
arm() { # $1 off|on  $2 label
    local a=$1 label=$2
    local log="$EV/$label-server.log" extra=""
    [ "$a" = on ] && extra="MEMRA_KV_HOST_CONTRACTS=1"
    mark "$label boot"
    boot "$extra" "$log" || { mark "$label boot-failed"; return 1; }
    mark "$label ready"
    local i=1 p
    for p in A B A B A B A; do
        if [ "$p" = A ]; then req "$P_A" "$EV/$label-r$i.json"; else req "$P_B" "$EV/$label-r$i.json"; fi
        mark "$label r$i done"
        i=$((i+1))
    done
    stop
    mark "$label stopped"
    # The door's own lines for this boot (the census the reading script counts), and the boot lines that
    # name the door and the tier.
    grep -E 'submitted off the tick|published off the tick|landed off the tick|request parked|refused|DISABLED|dropped|D2H receipt|H2D receipt' "$log" > "$EV/$label-door-lines.txt" || true
    grep -E '^\[prefix-host\] (contracts door|on:)|^\[prefix-cache\] on:|^\[server\] build:' "$log" > "$EV/$label-boot-lines.txt" || true
    echo "$label: $(grep -c '\[prefix-host\] demote: ' "$log") demotes, $(grep -c '\[prefix-host\] promote: ' "$log") promotes, $(grep -c 'request parked' "$log") parked, $(grep -c 'restore submitted off the tick' "$log") restores"
}
: > "$EV/marks.tsv"
# This script's own 1 s sampler across the four boots (the collector samples at 250 ms beside it).
nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used,utilization.gpu \
    --format=csv -lms 1000 > "$EV/card.during.csv" 2> "$EV/card.during.err" &
SAMPLER=$!
rc=0
arm off o1-off || rc=1
arm on  o1-on  || rc=1
arm on  o2-on  || rc=1
arm off o2-off || rc=1
stop_sampler
snap after
echo "wc-pair cell rc=$rc"
exit $rc
