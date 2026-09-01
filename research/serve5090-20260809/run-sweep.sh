#!/usr/bin/env bash
# Local 5090 q27 serve-surface floor receipt. One lock hold, fresh server per arm/rep.
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

LANE=research/serve5090-20260809
RAW=$LANE/raw
TS=${SERVE5090_TS:-$(date -u +%Y%m%dT%H%M%SZ)}
DRIVER=$RAW/driver-$TS.log
POINTS=$RAW/decode-points-$TS.jsonl
REQUESTS=$RAW/decode-requests-$TS.jsonl
CACHE_POINTS=$RAW/cache-ttft-$TS.jsonl
GPU=$RAW/gpu-$TS.csv
PORT=${SERVE5090_PORT:-8189}
REPS=${SERVE5090_REPS:-3}
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
BIN=$ROOT/target/release/memra-server
MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
SHORT=$ROOT/research/serve-ready-20260808/raw/short-prompt.txt
PROMPT4K=$ROOT/research/step-sku-20260807/prompt-pp4096.txt

mkdir -p "$RAW"
exec > >(tee "$DRIVER") 2>&1

cleanup_pid=
stop_server() {
    local pid=${1:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server $pid missed graceful-stop deadline"
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    return 1
}
trap 'stop_server "$cleanup_pid"' EXIT INT TERM

gpu_sample() {
    {
        printf '%s,' "$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,memory.used,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate \
            --format=csv,noheader | paste -sd ';' -
    } >> "$GPU"
}

port_free() {
    ! ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"
}

wait_up() {
    local pid=$1
    for _ in $(seq 1 240); do
        if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
            return 0
        fi
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

wait_worker_idle() {
    for _ in $(seq 1 30); do
        if curl -sf "$BASE/metrics" | python3 -c \
            'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("serve_idle_seconds",0)>=0.5 else 1)' \
            2>/dev/null; then
            return 0
        fi
        sleep 0.2
    done
    return 1
}

check_load_point() {
    python3 - "$1" <<'PY'
import json, pathlib, sys
lines = [line for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line.strip()]
point = json.loads(lines[-1])
assert point["n_ok"] == point["requests"], point
assert point["n_err"] == 0 and point["n_shed"] == 0, point
assert point.get("n_ttft") == point["n_ok"], point
PY
}

developer_probe() {
    local label=$1 out=$2
    curl -sf -m 300 "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"q27\",\"messages\":[{\"role\":\"developer\",\"content\":\"Answer tersely and accurately.\"},{\"role\":\"user\",\"content\":\"Reply with the word ready.\"}],\"max_tokens\":8,\"temperature\":0,\"cache_salt\":\"$label\"}" \
        | tee "$out"
    python3 - "$out" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
assert row["usage"]["completion_tokens"] > 0, row
assert row["choices"][0]["finish_reason"] in ("length", "stop"), row
PY
}

c_order() {
    case "$1" in
        1) echo "1 2 4" ;;
        2) echo "4 2 1" ;;
        *) echo "2 4 1" ;;
    esac
}

run_arm() {
    local arm=$1 rep=$2
    local label="${arm}-r${rep}"
    local shared_salt="q27-${label}-shared"
    local server_log=$RAW/server-$label-$TS.log
    local -a policy command
    case "$arm" in
        onpolicy) policy=() ;;
        specoff) policy=(MEMRA_SERVE_SPEC=0) ;;
        *) echo "FAIL: unknown arm $arm"; return 2 ;;
    esac

    port_free || {
        echo "FAIL: port $PORT already listening before $label"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        return 1
    }
    command=(
        env
        -u MEMRA_SERVE_SPEC
        -u MEMRA_SPEC_K
        -u MEMRA_SPEC_GATE
        -u MEMRA_SPEC_GATE_LOW
        -u MEMRA_SPEC_GATE_HIGH
        -u MEMRA_PREFIX_DEDUP
        -u MEMRA_PRIME_CHUNK
        -u MEMRA_PRIME_CHUNK_SCHED
        -u MEMRA_SERVE_BATCH
        -u MEMRA_DECODE_BATCH_CAP
        -u MEMRA_API_KEY
        -u MEMRA_API_KEYS
        CUDA_VISIBLE_DEVICES=0
        "${policy[@]}"
        MEMRA_MODELS="q27=${MODEL}+${DRAFT}"
        MEMRA_ADDR="$ADDR"
        MEMRA_COMPAT=openai
        MEMRA_CTX=8192
        MEMRA_MAX_SESSIONS=4
        MEMRA_REUSE_POOL=1
        MEMRA_PREFIX_CACHE_MB=512
        "$BIN"
    )

    echo "=== arm $label ==="
    printf 'command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    "${command[@]}" > "$server_log" 2>&1 &
    cleanup_pid=$!
    if ! wait_up "$cleanup_pid"; then
        echo "FAIL: $label server did not become ready"
        tail -100 "$server_log" || true
        return 1
    fi
    if ! ss -tlnp 2>/dev/null | grep -E "[:.]${PORT}[[:space:]]" \
            | grep -q "pid=$cleanup_pid,"; then
        echo "FAIL: $label port responder is not child pid $cleanup_pid"
        ss -tlnp | grep -E "[:.]${PORT}[[:space:]]" || true
        return 1
    fi
    echo "server ready pid=$cleanup_pid $(date -u +%FT%TZ)"
    gpu_sample

    if [[ $arm == specoff ]]; then
        echo "--- exact four-request prefix dedup/pinning + metering gate (fresh metrics) ---"
        python3 tools/cache-meter-gate.py "$BASE" q27 --n 4 --k 256 --suffix 16 \
            --raw-out "$RAW/cache-meter-$label-$TS.jsonl" \
            > "$RAW/cache-meter-$label-$TS.log" 2>&1
        sed -n '1,120p' "$RAW/cache-meter-$label-$TS.log"
        gpu_sample
    fi

    echo "--- developer-role smoke ---"
    developer_probe "$shared_salt" "$RAW/developer-$label-$TS.json"

    echo "--- cold short TTFT: one warmup + one scored row ---"
    python3 "$LANE/cold_ttft.py" \
        --base "$BASE" --model q27 --shape short --prompt-file "$SHORT" \
        --cache-salt "$shared_salt" --label "q27-$label" \
        --out "$RAW/ttft-short-$label-$TS.jsonl" --timeout 900
    gpu_sample

    echo "--- cold 4k TTFT: one warmup + one scored row ---"
    python3 "$LANE/cold_ttft.py" \
        --base "$BASE" --model q27 --shape 4k --prompt-file "$PROMPT4K" \
        --cache-salt "$shared_salt" --label "q27-$label" \
        --out "$RAW/ttft-4k-$label-$TS.jsonl" --timeout 900
    gpu_sample

    echo "--- exact-repeat 4k TTFT ---"
    python3 "$LANE/cached_ttft.py" \
        --base "$BASE" --model q27 --prompt-file "$PROMPT4K" \
        --cache-salt "$shared_salt" --label "q27-$label" --out "$CACHE_POINTS" \
        --mode repeat --expect-spec "$([[ $arm == onpolicy ]] && echo on || echo off)"
    gpu_sample

    echo "--- cached 4k continuation TTFT ---"
    python3 "$LANE/cached_ttft.py" \
        --base "$BASE" --model q27 --prompt-file "$PROMPT4K" \
        --cache-salt "$shared_salt" --label "q27-$label" --out "$CACHE_POINTS" \
        --mode continuation --expect-spec "$([[ $arm == onpolicy ]] && echo on || echo off)"
    gpu_sample

    echo "--- decode ladder (pair-receipt protocol) ---"
    local concurrency load_log
    for concurrency in $(c_order "$rep"); do
        load_log=$RAW/decode-$label-c$concurrency-$TS.log
        python3 tools/load-serve.py \
            --base "$BASE" --model q27 --concurrency "$concurrency" \
            --stream --max-tokens 128 --warmup 1 \
            --label "q27-$label-c$concurrency" \
            --out "$POINTS" --per-request "$REQUESTS" \
            > "$load_log" 2>&1
        sed -n '1,5p' "$load_log"
        check_load_point "$load_log"
        gpu_sample
    done

    curl -sf "$BASE/metrics" > "$RAW/metrics-$label-$TS.json"
    wait_worker_idle || {
        echo "FAIL: $label worker did not become idle before shutdown"
        return 1
    }
    stop_server "$cleanup_pid"
    cleanup_pid=
    sleep 3
    gpu_sample

    if grep -nEi 'CUDA_ERROR|out of memory|illegal address|spec pending flush failed|thread .*panicked|fatal signal' \
            "$server_log"; then
        echo "FAIL: $label server log contains a fatal signature"
        return 1
    fi
    if [[ $arm == onpolicy ]]; then
        grep -qE '\[spec-k\] model=.* K=3 source=cold-short' "$server_log" \
            || { echo "FAIL: $label missing cold-short K=3 receipt"; return 1; }
        grep -qE '\[spec-k\] model=.* K=3 source=cold-long' "$server_log" \
            || { echo "FAIL: $label missing cold-long K=3 receipt"; return 1; }
        grep -qE '\[spec-k\] model=.* K=2 source=cached-long' "$server_log" \
            || { echo "FAIL: $label missing cached-long K=2 receipt"; return 1; }
    else
        if grep -qE '\[spec-k\] model=.* K=[1-9]' "$server_log"; then
            echo "FAIL: $label ran nonzero K under MEMRA_SERVE_SPEC=0"
            return 1
        fi
        grep -qE '\[prefix-dedup\] B=3 .*retained=true' "$server_log" \
            || { echo "FAIL: $label missing retained three-follower prefix-dedup receipt"; return 1; }
    fi
    echo "arm complete $label $(date -u +%FT%TZ)"
}

for artifact in "$BIN" "$MODEL" "$DRAFT" "$SHORT" "$PROMPT4K"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
test -x "$BIN"

echo "=== serve5090 floor sweep $TS ==="
echo "measurement_commit=$(git rev-parse HEAD) engine_tip=96a09705895af120a0f706558a8c8c0d6fd8520a"
echo "branch=$(git branch --show-current) hostname=$(hostname)"
git status --short
echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
echo "model=$(stat -c '%n %s bytes' "$MODEL")"
echo "draft=$(stat -c '%n %s bytes' "$DRAFT")"
sha256sum "$SHORT" "$PROMPT4K"
echo "window_clean=false reason=owner-approved Hermes idle CUDA context"
echo "platform_profile=$(sed -n '1p' /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)"
echo "nv_dynamic_boost=$(sed -n '1p' /sys/devices/virtual/firmware-attributes/asus-armoury/attributes/nv_dynamic_boost/current_value 2>/dev/null || echo unknown)"
echo "nv_tgp=$(sed -n '1p' /sys/devices/virtual/firmware-attributes/asus-armoury/attributes/nv_tgp/current_value 2>/dev/null || echo unknown)"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader

exec 9>/tmp/gpu5090.lock
flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "GPU lock acquired $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$RAW/gpu-processes-pre-$TS.csv" 2>&1 || true
gpu_sample

for rep in $(seq 1 "$REPS"); do
    if ((rep % 2 == 1)); then
        run_arm onpolicy "$rep"
        run_arm specoff "$rep"
    else
        run_arm specoff "$rep"
        run_arm onpolicy "$rep"
    fi
done

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$RAW/gpu-processes-post-$TS.csv" 2>&1 || true
gpu_sample
echo "GPU lock released $(date -u +%FT%TZ)"
flock -u 9

echo "SERVE5090_SWEEP_DONE ts=$TS"
