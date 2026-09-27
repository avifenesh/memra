#!/usr/bin/env bash
# Day 25 retire-seam settle cell on the target card (DAY25.md pre-registration): four boots inside ONE
# collector lock hold, door OFF, ON, ON, OFF (pass 1 = off1/on1, pass 2 = on2/off2, A's day-20 order), each
# boot running day25-retire-cell.py twice, arm `plain` (the shape as briefed) then arm `coincide` (the seam
# exercised by a victim's abort in the seed's iteration), N=5 per arm per order, both orders. Boot: lane A's
# day-16 shape (`MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_KV_HOST_MB=8192`) with the prefix
# budget at 8192 MB instead of 1024 so the twenty-one 5120/5184-token entries (about 311 MB each on the 27B)
# of one boot never evict: an eviction demote is Move 1's term (ON 228 ms against OFF 70 ms in A's day-20
# receipts) and would swamp a 3 ms clause on the tenant's stall; this cell isolates the settle. The collector
# holds the lock and samples the card at 250 ms; the reading runs at the end and writes verdict.txt. Every
# cell is executed-not-qualified development evidence. No host, id or price here.
# usage: day25-retire-cell.sh <lockfd> <tree> <receipts_root> <model.gguf> <server_bin>
set -uo pipefail
fd=$1; TREE=$2; R=$3; MODEL=$4; BIN=$5
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
cd "$TREE" || exit 1
EV=$R/retire/ev
mkdir -p "$EV"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum "$BIN" | tee "$EV/binary.sha256"
{
    echo "tree=$(git rev-parse HEAD)"
    echo "harness_sha256=$(sha256sum research/spill-c-20260919/day25-retire-cell.py | cut -d' ' -f1)"
    echo "harness_a_sha256=$(sha256sum research/spill-a-20260919/stall_cell.py | cut -d' ' -f1)"
    echo "model=$(basename "$MODEL")"
    echo "gpu=$(nvidia-smi --query-gpu=name,power.limit --format=csv,noheader | head -1)"
    echo "boots=off1 on1 on2 off2"
    echo "arms=plain coincide"
    echo "boot_env=MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 MEMRA_SERVE_SPEC=0 MEMRA_PREFIX_CACHE_MB=8192 MEMRA_KV_HOST_MB=8192 (+MEMRA_KV_HOST_CONTRACTS=1 on the ON boots)"
    echo "status=executed-not-qualified"
} > "$EV/CELL.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.before.csv" 2>&1
PORT=${MEMRA_GATE_PORT:-18135}
. tools/port-guard.sh
SERVER_PID=""
boot() { # $1 extra-env-string  $2 log
    memra_port_guard retire-cell "$PORT" MEMRA_GATE_PORT || return 1
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
H=research/spill-c-20260919/day25-retire-cell.py
: > "$EV/marks.tsv"
: > "$EV/replays.log"
rc=0
for b in off1 on1 on2 off2; do
    D=$EV/$b; mkdir -p "$D"
    extra="MEMRA_PREFIX_CACHE_MB=8192 MEMRA_KV_HOST_MB=8192"
    [[ $b == on* ]] && extra="$extra MEMRA_KV_HOST_CONTRACTS=1"
    mark "boot-$b"; boot "$extra" "$D/server.log" || { rc=1; echo "boot $b failed" | tee -a "$EV/replays.log"; break; }; mark "ready-$b"
    for arm in plain coincide; do
        python3 $H --port "$PORT" --server-log "$D/server.log" --out "$D/$arm" --tag "retire-$b" --arm "$arm" --n 5 | tee "$D/$arm.log" || rc=1
        mark "$arm-done-$b"
    done
    stop; mark "stopped-$b"
    for arm in plain coincide; do
        { echo "== $b/$arm"; python3 $H --replay "$D/$arm/receipt.json"; } | tee -a "$EV/replays.log" || rc=1
    done
done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$EV/compute-apps.after.csv" 2>&1
if [ $rc -eq 0 ]; then
    python3 research/spill-c-20260919/day25-retire-reading.py "$EV" | tee "$EV/reading.log"
    grep "DAY25 RETIRE VERDICT" "$EV/reading.log" > "$EV/verdict.txt"
else
    echo "DAY25 RETIRE VERDICT: not read (a boot or a run failed, rc=$rc)" | tee "$EV/verdict.txt"
fi
echo "retire rc=$rc" | tee "$EV/exit.txt"
exit $rc
