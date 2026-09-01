#!/usr/bin/env bash
# One clean Q27 server boot: seed an identical-prefix set, then hit those snapshots.
set -euo pipefail

cd "$(dirname "$0")/../.."

LABEL=${1:?usage: $0 LABEL SEED_CONCURRENCY HIT_CONCURRENCY [SEED_COUNT] [HIT_REPETITIONS]}
SEED_CONCURRENCY=${2:?missing seed concurrency}
HIT_CONCURRENCY=${3:?missing hit concurrency}
SEED_COUNT=${4:-8}
HIT_REPETITIONS=${5:-1}
PORT=${EOSCLASS_PORT:-18427}
BASE=http://127.0.0.1:$PORT
MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
SERVER=target/release/memra-server
OUT=research/eosclass-20260813/raw/$LABEL
LOCK=/tmp/memra-5090.lock
LOCK_WAIT=${EOSCLASS_LOCK_WAIT_SECONDS:-0}
CLIENT_ARM=${EOSCLASS_ARM:-seed-hits}

test -f "$MODEL"
test -x "$SERVER"
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 2; }

exec 9>"$LOCK"
if test "$LOCK_WAIT" != 0; then
    echo "waiting up to ${LOCK_WAIT}s for GPU lease: $LOCK" >&2
fi
if ! flock -w "$LOCK_WAIT" 9; then
    echo "GPU lease busy or wait expired: $LOCK" >&2
    exit 75
fi
mkdir -p "$OUT"

SERVER_PID=
stop_server() {
    test -n "${SERVER_PID:-}" || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=
            return 0
        fi
        sleep 1
    done
    echo "owned server pid=$SERVER_PID did not stop after 60 seconds" >&2
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    return 1
}
trap stop_server EXIT

if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
    echo "port $PORT already has a listener" >&2
    exit 1
fi

{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "head=$(git rev-parse HEAD)"
    echo "tag=$(git describe --tags --exact-match HEAD 2>/dev/null || true)"
    echo "branch=$(git branch --show-current)"
    git status --short
    echo "model=$MODEL"
    sha256sum "$MODEL" "$SERVER" research/sellgate-20260812/sellgate_replay.py \
        research/sellgate-20260812/workload.lock.json \
        research/eosclass-20260813/repro_cache_seed.py \
        research/eosclass-20260813/repro_restore_mix.py \
        research/eosclass-20260813/repro_width_flip.py
    echo "seed_concurrency=$SEED_CONCURRENCY"
    echo "hit_concurrency=$HIT_CONCURRENCY"
    echo "seed_count=$SEED_COUNT"
    echo "hit_repetitions=$HIT_REPETITIONS"
    echo "client_arm=$CLIENT_ARM"
    echo "client_expect=${EOSCLASS_EXPECT:-observe}"
    echo "prime_batch=${EOSCLASS_PRIME_BATCH:-default}"
    echo "prime_batch_hold_ms=${EOSCLASS_PRIME_BATCH_HOLD_MS:-default}"
    echo "eosclass_trace=${EOSCLASS_TRACE:-0}"
    echo "serve_devsample=${EOSCLASS_DEVSAMPLE:-default}"
    echo "serve_b1fast=${EOSCLASS_B1FAST:-default}"
    echo "serve_gs=${EOSCLASS_GS:-default}"
    echo "cold_peers=${EOSCLASS_COLD_PEERS:-3}"
    echo "width_flip_delays_ms=${EOSCLASS_DELAYS_MS:-default}"
    echo "gpu_lock=$LOCK"
    echo "gpu_lock_wait_seconds=$LOCK_WAIT"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
} >"$OUT/provenance.log" 2>&1

server_env=(
    -u MEMRA_PP_STAGES
    -u MEMRA_PP_DEVICES
    -u MEMRA_DUAL_PP
    -u MEMRA_PP_OVERLAP
    -u MEMRA_PP_HOST_BOUNCE
    -u MEMRA_PRIME_PIPE
    -u MEMRA_PREFILL_TICK
    -u MEMRA_SERVE_BATCH
    -u MEMRA_SPEC_K
    -u MEMRA_SPEC_GATE
    -u MEMRA_DECODE_BATCH_CAP
    -u MEMRA_SERVE_B1FAST
    -u MEMRA_SERVE_GS
    -u MEMRA_PRIME_BATCH
    -u MEMRA_PRIME_BATCH_HOLD_MS
    -u MEMRA_SERVE_DEVSAMPLE
    -u MEMRA_FAST
    -u MEMRA_MOE_RESIDENT
    -u MEMRA_MOE_RESIDENT_GB
    CUDA_VISIBLE_DEVICES=0
    MEMRA_MODELS=q27-eosclass=$MODEL
    MEMRA_COMPAT=openai
    MEMRA_ADDR=127.0.0.1:$PORT
    MEMRA_TAG=cx-eosclass-$LABEL
    MEMRA_SERVE_SPEC=0
    MEMRA_CTX=8192
    MEMRA_PREFIX_CACHE_MB=4096
    MEMRA_PREFIX_DEDUP=1
    MEMRA_REUSE_POOL=0
    MEMRA_AFFINITY=0
    MEMRA_MAX_SESSIONS=96
    MEMRA_EOSCLASS_TRACE="${EOSCLASS_TRACE:-0}"
)
if test -n "${EOSCLASS_PRIME_BATCH:-}"; then
    server_env+=(MEMRA_PRIME_BATCH="$EOSCLASS_PRIME_BATCH")
fi
if test -n "${EOSCLASS_PRIME_BATCH_HOLD_MS:-}"; then
    server_env+=(MEMRA_PRIME_BATCH_HOLD_MS="$EOSCLASS_PRIME_BATCH_HOLD_MS")
fi
if test -n "${EOSCLASS_DEVSAMPLE:-}"; then
    server_env+=(MEMRA_SERVE_DEVSAMPLE="$EOSCLASS_DEVSAMPLE")
fi
if test -n "${EOSCLASS_B1FAST:-}"; then
    server_env+=(MEMRA_SERVE_B1FAST="$EOSCLASS_B1FAST")
fi
if test -n "${EOSCLASS_GS:-}"; then
    server_env+=(MEMRA_SERVE_GS="$EOSCLASS_GS")
fi

env "${server_env[@]}" "$SERVER" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$OUT/server.pid"
for _ in $(seq 1 900); do
    if curl -sf "$BASE/readyz" >"$OUT/readyz.json" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server died during boot" >&2
        tail -200 "$OUT/server.log" >&2
        exit 1
    fi
    sleep 1
done
curl -sf "$BASE/readyz" >"$OUT/readyz.json"
curl -sf "$BASE/metrics" >"$OUT/metrics-before.json"
nvidia-smi --query-gpu=index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader >"$OUT/gpu-ready.csv"

set +e
if test "$CLIENT_ARM" = restore-mix; then
    timeout 7200 python3 research/eosclass-20260813/repro_restore_mix.py \
        --base "$BASE" --model q27-eosclass --namespace "cx-eosclass-$LABEL" \
        --cold-peers "${EOSCLASS_COLD_PEERS:-3}" --repetitions "$HIT_REPETITIONS" \
        --expect "${EOSCLASS_EXPECT:-observe}" 2>&1 | tee "$OUT/client.jsonl"
elif test "$CLIENT_ARM" = seed-hits; then
    timeout 7200 python3 research/eosclass-20260813/repro_cache_seed.py \
        --base "$BASE" --model q27-eosclass --namespace "cx-eosclass-$LABEL" \
        --seed-count "$SEED_COUNT" --seed-concurrency "$SEED_CONCURRENCY" \
        --hit-concurrency "$HIT_CONCURRENCY" --hit-repetitions "$HIT_REPETITIONS" \
        --expect "${EOSCLASS_EXPECT:-observe}" 2>&1 | tee "$OUT/client.jsonl"
elif test "$CLIENT_ARM" = width-flip; then
    width_flip_args=(
        --base "$BASE" --model q27-eosclass --namespace "cx-eosclass-$LABEL"
        --peer-count "${EOSCLASS_COLD_PEERS:-3}" --repetitions "$HIT_REPETITIONS"
        --expect "${EOSCLASS_EXPECT:-observe}"
    )
    if test -n "${EOSCLASS_DELAYS_MS:-}"; then
        width_flip_args+=(--delays-ms "$EOSCLASS_DELAYS_MS")
    fi
    timeout 7200 python3 research/eosclass-20260813/repro_width_flip.py \
        "${width_flip_args[@]}" 2>&1 | tee "$OUT/client.jsonl"
else
    echo "unknown EOSCLASS_ARM=$CLIENT_ARM" >&2
    false
fi
CLIENT_RC=${PIPESTATUS[0]}
set -e
echo "$CLIENT_RC" >"$OUT/client.exit"

curl -sf "$BASE/metrics" >"$OUT/metrics-after.json"
nvidia-smi --query-gpu=index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader >"$OUT/gpu-after.csv"
grep -Ein 'out of memory|CUDA_ERROR|panic|fatal|illegal address|misaligned address' \
    "$OUT/server.log" >"$OUT/failure-signature-scan.log" || true
grep -E '\[prime-batch\]|\[prefix-cache\]' "$OUT/server.log" \
    >"$OUT/prime-cache-events.log" || true
grep -E '\[eosclass-(snapshot|restore|sample)\]' "$OUT/server.log" \
    >"$OUT/eosclass-trace.jsonl" || true

stop_server
trap - EXIT
exit "$CLIENT_RC"
