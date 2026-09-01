#!/usr/bin/env bash
# Run one isolated Q35 mixed-c=2 recorder block on the eu-west PRO 6000 pair.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 LABEL SERVER COMPAT(openai|native) DECODE(default|b1-batched|serial) EXPECT_CLEAN(0|1)" >&2
    exit 2
fi

LABEL=$1
SERVER=$2
COMPAT=$3
DECODE=$4
EXPECT_CLEAN=$5
case "$COMPAT" in openai|native) ;; *) echo "invalid compat: $COMPAT" >&2; exit 2 ;; esac
case "$DECODE" in default|b1-batched|serial) ;; *) echo "invalid decode mode: $DECODE" >&2; exit 2 ;; esac
case "$EXPECT_CLEAN" in 0|1) ;; *) echo "invalid EXPECT_CLEAN: $EXPECT_CLEAN" >&2; exit 2 ;; esac

ROOT=${Q35BUG_ROOT:-/opt/dl-image/nvme/cx-q35bug}
HARNESS=$ROOT/harness
RAW=$ROOT/raw
OUT=$RAW/$LABEL
MODEL=/opt/dl-image/nvme/cx-percard/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
PORT=${Q35BUG_PORT:-18535}
BASE=http://127.0.0.1:$PORT
REPETITIONS=${Q35BUG_REPETITIONS:-5}

test ! -e "$OUT" || { echo "refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1
exec 9>/tmp/memra-gpu.lock
echo "lock_wait_start=$(date -u +%FT%TZ)"
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "lock_acquired=$(date -u +%FT%TZ)"

server_pid=
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

test -x "$SERVER"
test -f "$MODEL"
test -f "$HARNESS/repro.py"
test -f "$HARNESS/workload.lock.json"
if ss -tln 2>/dev/null | grep -q "[:.]$PORT "; then
    echo "FAIL: port $PORT already has a listener"
    ss -tlnp 2>/dev/null | grep "[:.]$PORT " || true
    exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo "FAIL: compute applications already active after GPU lock acquisition"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    exit 1
fi

{
    echo "label=$LABEL"
    echo "started=$(date -u +%FT%TZ)"
    echo "compat=$COMPAT"
    echo "decode_mode=$DECODE"
    echo "expect_clean=$EXPECT_CLEAN"
    echo "repetitions=$REPETITIONS"
    echo "source_commit=${Q35BUG_SOURCE_COMMIT:-unknown}"
    sha256sum "$SERVER" "$MODEL" "$HARNESS/repro.py" "$HARNESS/workload.lock.json"
    nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit,memory.total,memory.used,memory.free,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
} >"$OUT/provenance.txt"
cp "$HARNESS/workload.lock.json" "$OUT/workload.lock.json"

server_env=(
    env
    -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP
    -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE
    -u MEMRA_SERVE_BATCH -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE
    -u MEMRA_DECODE_BATCH_CAP -u MEMRA_FAST -u MEMRA_MOE_RESIDENT
    -u MEMRA_MOE_RESIDENT_GB
)
server_env+=(-u MEMRA_STEP35_BATCH)
if [[ "$DECODE" == default ]]; then
    server_env+=(-u MEMRA_SERVE_B1FAST)
elif [[ "$DECODE" == b1-batched ]]; then
    server_env+=(MEMRA_SERVE_B1FAST=0)
else
    server_env+=(-u MEMRA_SERVE_B1FAST)
fi
server_env+=(
    CUDA_VISIBLE_DEVICES=1
    MEMRA_MODELS=q35=$MODEL
    MEMRA_COMPAT=$COMPAT
    MEMRA_ADDR=127.0.0.1:$PORT
    MEMRA_TAG=cx-q35bug-$LABEL
    MEMRA_SERVE_SPEC=0
    MEMRA_CTX=8192
    MEMRA_PREFIX_CACHE_MB=4096
    MEMRA_PREFIX_DEDUP=1
    MEMRA_REUSE_POOL=0
    MEMRA_AFFINITY=0
    MEMRA_MAX_SESSIONS=96
    MEMRA_TICK_TRACE=1
)
if [[ "$DECODE" == serial ]]; then
    server_env+=(MEMRA_SERVE_BATCH=0)
fi
"${server_env[@]}" "$SERVER" >"$OUT/server.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 900); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "FAIL: server exited during startup"
        tail -200 "$OUT/server.log"
        exit 1
    fi
    if curl -sf "$BASE/v1/models" >"$OUT/models.json.tmp"; then
        mv "$OUT/models.json.tmp" "$OUT/models.json"
        ready=1
        break
    fi
    sleep 1
done
test "$ready" = 1 || { echo "FAIL: server readiness timeout"; exit 1; }

before_rc=0
python_args=(
    "$HARNESS/repro.py"
    --base "$BASE"
    --model q35
    --compat "$COMPAT"
    --workload-lock "$HARNESS/workload.lock.json"
    --out "$OUT/repro.jsonl"
    --namespace "cx-q35bug-$LABEL"
    --repetitions "$REPETITIONS"
)
if [[ "$EXPECT_CLEAN" == 1 ]]; then
    python_args+=(--expect-clean)
fi
python3 "${python_args[@]}" >"$OUT/repro.stdout" 2>"$OUT/repro.stderr" || before_rc=$?
curl -sf "$BASE/metrics" >"$OUT/metrics-final.json"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader >"$OUT/compute-apps-before-stop.txt" || true

kill "$server_pid"
wait "$server_pid" || true
server_pid=
grep -E '"kind": "(cell|summary)"' "$OUT/repro.stdout" || true
if grep -Ein 'CUDA_ERROR|out of memory|panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|mismatches=[1-9]' "$OUT/server.log" \
    || grep -En 'MISMATCH' "$OUT/server.log"; then
    echo "FAIL: server log contains a fatal/error signature"
    exit 1
fi
if [[ "$before_rc" -ne 0 ]]; then
    echo "FAIL: repro.py exited $before_rc"
    exit "$before_rc"
fi
echo "completed=$(date -u +%FT%TZ)"
