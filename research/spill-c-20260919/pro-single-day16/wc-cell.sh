#!/usr/bin/env bash
# WC pair cell (lane/spill-c-20260919 day 16, research/spill-c-20260919/WC-DESTINATIONS.md): door OFF versus
# ON demote and promote wall times for the SAME two entries, N=5 per arm, BOTH orders (OFF,ON then ON,OFF),
# under ONE collector lock hold (one thermal window, the collector's 250 ms telemetry). Four server boots.
# Per boot: r1 P_A seeds E_A; r2 P_B seeds E_B and evicts E_A (demote 1); r3..r7 alternate P_A/P_B: each
# is a host hit that promotes and whose insert evicts the other entry (demote inside the promote window).
# Demotes r2..r6 (N=5) and promotes r3..r7 (N=5) are the observations; the server log is the receipt.
# MEMRA_KV_HOST_VERIFY unset (the production promote shape; the verify digest is its own D2H readback).
# usage: wc-cell.sh <lockfd> [cell-name]   (the first sitting, `wc-pair`, was refused by an unbound variable in
# this script's own `local` line before any boot; the cell ran as `wc-pair2`)
set -uo pipefail
fd=$1
cell=${2:-wc-pair}
R=/root/spill-receipts/c-day16
EV=$R/$cell/ev
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum $R/bins/memra-server | tee "$EV/binary.sha256"
MODEL=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
BIN=$R/bins/memra-server
PORT=${MEMRA_GATE_PORT:-18119}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 extra-env-string  $2 log
    memra_port_guard wc-cell "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving, refusing to boot over it"; return 1
    fi
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        MEMRA_PREFIX_CACHE_MB=256 MEMRA_KV_HOST_MB=8192 $1 "$BIN" >"$2" 2>&1 &
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
    echo "$label: $(grep -c '\[prefix-host\] demote: ' "$log") demotes, $(grep -c '\[prefix-host\] promote: ' "$log") promotes"
}
: > "$EV/marks.tsv"
rc=0
arm off o1-off || rc=1
arm on  o1-on  || rc=1
arm on  o2-on  || rc=1
arm off o2-off || rc=1
echo "wc-pair cell rc=$rc"
exit $rc
