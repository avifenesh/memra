#!/usr/bin/env bash
# Bounded Vast serve-home anatomy block. No flock exists on this host: stop the production
# server and soak, run the diagnostic server/probe, then restore both even on failure.
set -euo pipefail

ROOT=${P2PVAST_ROOT:-/workspace/memra}
OUT=${P2PVAST_OUT:-/root/p2pvast-receipts/block2}
BASE=http://127.0.0.1:8002
MODEL=/workspace/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
MODEL_ID=stepfun/step-3.7-flash
PROMPT_4K=${ROOT}/research/step-sku-20260807/prompt-pp4096.txt
PROMPT_512=${ROOT}/research/chunk-invariance-20260805/prompt-pp512.txt
BIN=${ROOT}/target/release/memra-server

mkdir -p "$OUT"
cd "$ROOT"
exec > >(tee "$OUT/driver.log") 2>&1

diag_pid=
restore_started=0

stop_pid() {
    local pid=$1
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

stop_runtime() {
    pkill -f 'release/memra-serve[r]' 2>/dev/null || true
    pkill -f '[s]oak.py' 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! pgrep -f 'release/memra-serve[r]|[s]oak.py' >/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

restore_runtime() {
    if [[ "$restore_started" == 1 ]]; then
        return 0
    fi
    restore_started=1
    echo "RESTORE_BEGIN $(date -u +%FT%TZ)"
    if [[ -n "$diag_pid" ]]; then
        stop_pid "$diag_pid"
        diag_pid=
    fi
    stop_runtime || true
    cd "$ROOT"
    setsid nohup /root/start-memra.sh > /var/log/memra-server.log 2>&1 < /dev/null &
    disown || true
    sleep 90
    setsid nohup python3 /root/soak.py > /dev/null 2>&1 < /dev/null &
    disown || true
    sleep 2
    if ! curl -fsS "$BASE/v1/models" > "$OUT/restart-models.json"; then
        echo "RESTORE_FAIL: /v1/models did not answer"
        tail -n 160 /var/log/memra-server.log > "$OUT/restart-server-tail.log" 2>&1 || true
        return 1
    fi
    pgrep -af '[m]emra-server|[s]oak.py' > "$OUT/restart-processes.log"
    cp /var/log/memra-server.log "$OUT/restart-server.log"
    grep -E '\[pp\].*(cross-device transport|peer|grant)' /var/log/memra-server.log \
        > "$OUT/restart-peer-lines.log" || true
    echo "RESTORE_OK $(date -u +%FT%TZ)"
}

cleanup() {
    local rc=$?
    set +e
    restore_runtime
    local restore_rc=$?
    trap - EXIT
    if [[ "$rc" == 0 && "$restore_rc" != 0 ]]; then
        rc=$restore_rc
    fi
    exit "$rc"
}
trap cleanup EXIT

wait_ready() {
    local pid=$1
    for _ in $(seq 1 180); do
        if curl -fsS "$BASE/readyz" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            return 1
        fi
        sleep 2
    done
    return 1
}

request() {
    local label=$1
    local max_tokens=$2
    local prompt_file=${3:-}
    local request_json="$OUT/${label}-request.json"
    python3 - "$MODEL_ID" "$max_tokens" "$prompt_file" > "$request_json" <<'PY'
import json
import sys

model, max_tokens, prompt_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
if prompt_file:
    with open(prompt_file, encoding="utf-8") as handle:
        content = handle.read()
else:
    content = "Reply with exactly four words about peer-copy correctness."
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": max_tokens,
    "temperature": 0.0,
    "stream": False,
}))
PY
    curl --fail --silent --show-error --max-time 900 \
        -H 'Content-Type: application/json' \
        --data-binary "@$request_json" \
        -o "$OUT/${label}-response.json" \
        -w 'http_code=%{http_code}\ntime_total_s=%{time_total}\nsize_download=%{size_download}\n' \
        "$BASE/v1/chat/completions" > "$OUT/${label}-client.log" 2>&1
    python3 - "$OUT/${label}-response.json" > "$OUT/${label}-response-summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    response = json.load(handle)
print(json.dumps({key: response.get(key) for key in ("id", "model", "choices", "usage")},
                 indent=2, sort_keys=True))
PY
}

for artifact in "$MODEL" "$PROMPT_4K" "$PROMPT_512" "$BIN"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
source "$HOME/.cargo/env"
source /root/serve-env.sh
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/lib64:/usr/local/cuda-12.8/lib64

echo "BLOCK2_BEGIN $(date -u +%FT%TZ)"
git rev-parse HEAD > "$OUT/source-commit.txt"
git status --short > "$OUT/source-status.txt"
curl -fsS "$BASE/v1/models" > "$OUT/pre-models.json"
pgrep -af '[m]emra-server|[s]oak.py' > "$OUT/pre-processes.log" || true
sha256sum "$PROMPT_4K" "$PROMPT_512" > "$OUT/prompt-sha256.txt"
wc -c -w "$PROMPT_4K" "$PROMPT_512" > "$OUT/prompt-size.txt"
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-pre.csv"

stop_runtime
echo "RUNTIME_STOPPED $(date -u +%FT%TZ)"
pgrep -af '[m]emra-server|[s]oak.py' > "$OUT/processes-stopped.log" || true

cargo build --release -p memra-engine --bin concat-prime-probe \
    > "$OUT/build-prime-probe.log" 2>&1
sha256sum "$BIN" ./target/release/concat-prime-probe > "$OUT/binary-sha256.txt"

env \
    MEMRA_SPEC_GATE=0 \
    MEMRA_SPEC_K=1 \
    MEMRA_SPEC_STATS=1 \
    MEMRA_SPEC_PP_ANATOMY=1 \
    MEMRA_TTFT_TRACE=1 \
    "$BIN" > "$OUT/server.log" 2>&1 &
diag_pid=$!
if ! wait_ready "$diag_pid"; then
    echo "FAIL: diagnostic server did not become ready"
    tail -n 200 "$OUT/server.log" || true
    exit 1
fi
boot_end=$(wc -l < "$OUT/server.log")
echo "$boot_end" > "$OUT/server-boot-end-line.txt"

echo "SHORT_BEGIN $(date -u +%FT%TZ)"
request short 64
sleep 1
short_end=$(wc -l < "$OUT/server.log")
echo "$short_end" > "$OUT/server-short-end-line.txt"
echo "SHORT_DONE $(date -u +%FT%TZ)"

echo "LONG4K_BEGIN $(date -u +%FT%TZ)"
request long4k 8 "$PROMPT_4K"
sleep 1
long_end=$(wc -l < "$OUT/server.log")
echo "$long_end" > "$OUT/server-long4k-end-line.txt"
echo "LONG4K_DONE $(date -u +%FT%TZ)"

curl -fsS "$BASE/metrics" > "$OUT/metrics.txt"
sed -n "1,${boot_end}p" "$OUT/server.log" > "$OUT/server-boot.log"
sed -n "$((boot_end + 1)),${short_end}p" "$OUT/server.log" > "$OUT/server-short.log"
sed -n "$((short_end + 1)),${long_end}p" "$OUT/server.log" > "$OUT/server-long4k.log"
grep -E '\[spec-(pp-)?anatomy\]|\[spec-phase\]|\[spec-stats\]|\[ttft\]' \
    "$OUT/server-short.log" > "$OUT/short-anatomy-lines.log" || true
grep -E '\[spec-(pp-)?anatomy\]|\[spec-phase\]|\[spec-stats\]|\[ttft\]' \
    "$OUT/server-long4k.log" > "$OUT/long4k-anatomy-lines.log" || true
grep -E -i 'error|illegal|sentinel|panic|abort|CUDA_ERROR' "$OUT/server.log" \
    > "$OUT/server-error-scan.log" || true

stop_pid "$diag_pid"
diag_pid=
echo "DIAGNOSTIC_SERVER_STOPPED $(date -u +%FT%TZ)"

set +e
env \
    MEMRA_PP_STAGES=2 \
    MEMRA_PP_DEVICES=0,1 \
    MEMRA_MOE_GROUPED=1 \
    MEMRA_PRIME_CHUNK_SCHED=dynamic \
    ./target/release/concat-prime-probe "$MODEL" pppipeperf \
        --prompt-a "@$PROMPT_512" --reps 1 --warmup 0 \
        > "$OUT/prime-pipeline-diagnostic.log" 2>&1
prime_rc=$?
set -e
echo "prime_pipeline_diagnostic=$prime_rc" > "$OUT/exit-codes.txt"

nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu,clocks.sm,power.draw \
    --format=csv,noheader > "$OUT/gpu-post.csv"
echo "BLOCK2_MEASUREMENT_DONE $(date -u +%FT%TZ)"

restore_runtime
trap - EXIT
echo "BLOCK2_DONE $(date -u +%FT%TZ)"
