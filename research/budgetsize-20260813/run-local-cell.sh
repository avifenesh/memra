#!/usr/bin/env bash
# One fail-closed local Q27 serving cell. The caller alternates baseline/derived repetitions
# and runs the explicit-budget and c=64 checks separately.
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 BINARY baseline|derived|explicit4096-a|explicit4096-b|derived-c64|explicit4096-a-c64 REP OUT_DIR" >&2
    exit 2
fi

BIN=$1
ARM=$2
REP=$3
OUT=$4
case "$ARM" in
    baseline|explicit4096-a|explicit4096-a-c64) ARM_KEY=A ;;
    derived|explicit4096-b|derived-c64) ARM_KEY=B ;;
    *) echo "invalid arm: $ARM" >&2; exit 2 ;;
esac

REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
PROTOCOL=$REPO/research/budgetsize-20260813/protocol.lock.json
REPLAY=$REPO/research/budgetsize-20260813/replay.py
PORT=18460
BASE=http://127.0.0.1:$PORT
SERVER_PID=
SAMPLER_PID=

test -x "$BIN"
test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 1; }
if [ "${MEMRA_5090_LOCK_HELD:-0}" != 1 ]; then
    exec 9>/tmp/memra-5090.lock
    flock -n 9 || { echo "FAIL: /tmp/memra-5090.lock is held" >&2; exit 75; }
fi
test "$(sha256sum "$MODEL" | awk '{print $1}')" = d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
python3 -m json.tool "$PROTOCOL" >/dev/null
EXPECTED_BIN_SHA=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["arms"][sys.argv[2]]["binary_sha256"])' \
    "$PROTOCOL" "$ARM_KEY")
ACTUAL_BIN_SHA=$(sha256sum "$BIN" | awk '{print $1}')
test "$ACTUAL_BIN_SHA" = "$EXPECTED_BIN_SHA" || {
    echo "FAIL: arm $ARM requires binary $EXPECTED_BIN_SHA, got $ACTUAL_BIN_SHA" >&2
    exit 1
}
python3 -m py_compile "$REPLAY"
bash -n "$0"

mkdir -p "$OUT"
exec > >(tee "$OUT/orchestrator.log") 2>&1

stop_server() {
    test -n "${SERVER_PID:-}" || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=
            return 0
        fi
        sleep 1
    done
    echo "FAIL: owned server did not stop after 120 seconds"
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    return 1
}

stop_sampler() {
    test -n "${SAMPLER_PID:-}" || return 0
    kill "$SAMPLER_PID" 2>/dev/null || true
    wait "$SAMPLER_PID" 2>/dev/null || true
    SAMPLER_PID=
}

cleanup() {
    stop_server || true
    stop_sampler
}
trap cleanup EXIT INT TERM

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
    echo "FAIL: a compute process exists after acquiring the 5090 lock"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    exit 1
fi

{
    echo "arm=$ARM repetition=$REP"
    echo "ts=$(date -u +%FT%TZ)"
    echo "binary=$BIN"
    sha256sum "$BIN" "$MODEL" "$REPLAY" "$REPO/research/sellgate-20260812/sellgate_replay.py" \
        "$REPO/research/sellgate-20260812/workload.lock.json"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,persistence_mode,pstate,temperature.gpu,\
clocks.current.sm,memory.total,memory.used,memory.free --format=csv,noheader
} >"$OUT/preflight.log" 2>&1

nvidia-smi --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,\
clocks.current.sm,clocks.current.memory,memory.total,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu-250ms.csv" 2>&1 &
SAMPLER_PID=$!

budget_env=()
case "$ARM" in
    explicit4096-a|explicit4096-b|explicit4096-a-c64)
        budget_env=(MEMRA_PREFIX_CACHE_MB=4096)
        ;;
esac
env -u MEMRA_PREFIX_CACHE_MB -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
    -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
    -u MEMRA_PRIME_BATCH -u MEMRA_PREFILL_TICK -u MEMRA_SERVE_BATCH \
    -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE -u MEMRA_DECODE_BATCH_CAP \
    -u MEMRA_FAST -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
    "${budget_env[@]}" CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="q27=$MODEL" \
    MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_TAG="cx-budgetsize-$ARM-r$REP" \
    MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 MEMRA_PREFIX_DEDUP=1 MEMRA_REUSE_POOL=0 \
    MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=64 "$BIN" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 900); do
    if curl -sf "$BASE/readyz" >/dev/null 2>&1; then ready=1; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died during boot"
        tail -200 "$OUT/server.log"
        exit 1
    fi
    sleep 1
done
test "$ready" -eq 1 || { echo "FAIL: server readiness timeout"; exit 1; }
curl -sf "$BASE/v1/models" >"$OUT/models.json"
curl -sf "$BASE/metrics" >"$OUT/metrics-ready.json"

set +e
timeout 3600 python3 "$REPLAY" --endpoint "$BASE" --arm "$ARM" --repetition "$REP" \
    --namespace "cx-budgetsize-$ARM-r$REP" --protocol "$PROTOCOL" \
    --out "$OUT/replay.jsonl" --timeout 1800 2>&1 | tee "$OUT/replay.log"
replay_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$replay_rc" >"$OUT/replay.exit"
curl -sf "$BASE/metrics" >"$OUT/metrics-final.json"
stop_server
stop_sampler

nvidia-smi --query-gpu=index,name,pstate,temperature.gpu,clocks.current.sm,memory.total,\
memory.used,memory.free --format=csv,noheader >"$OUT/gpu-final.log" 2>&1
awk -F, '{sm_mhz=$7 + 0; if (sm_mhz < 210 || sm_mhz > 1200) bad=1} END {exit bad}' \
    "$OUT/gpu-250ms.csv" || { echo "FAIL: SM clock sample escaped 210-1200 MHz"; exit 1; }
grep -aEin 'panicked at|worker.*died|server.*FATAL|illegal memory access|CUDA_ERROR|out of memory' \
    "$OUT/server.log" >"$OUT/server-failure-scan.log" || true
test ! -s "$OUT/server-failure-scan.log"
test "$replay_rc" -eq 0
grep -q '"verdict": "PASS"' "$OUT/replay.jsonl"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"
echo "CELL_PASS arm=$ARM repetition=$REP ts=$(date -u +%FT%TZ)"
