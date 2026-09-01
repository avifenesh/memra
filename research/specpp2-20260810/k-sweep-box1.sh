#!/usr/bin/env bash
# N=5 interleaved c=1 plain versus forced speculative K=1/2/3 on box1.
set -euo pipefail

ROOT=${SPEC_PP2_ROOT:-/home/ubuntu/memra-cx-specpp2}
OUT=${SPEC_PP2_OUT:-/home/ubuntu/specpp2-receipts/k-sweep}
PORT=${SPEC_PP2_PORT:-8139}
ADDR=127.0.0.1:${PORT}
BASE=http://${ADDR}
MODEL=/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=/home/ubuntu/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
BIN=${ROOT}/target/release/memra-server

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
exec > >(tee "$OUT/driver.log") 2>&1

for artifact in "$MODEL" "$DRAFT" "$BIN"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

wait_up() {
    local pid=$1
    for _ in $(seq 1 360); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

stop_server() {
    local pid=$1
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            return
        fi
        sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

gpu_sample() {
    {
        printf '%s,' "$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm,power.draw \
            --format=csv,noheader | paste -sd ';' -
    } >> "$OUT/gpu-samples.csv"
}

server_pid=
trap 'test -z "$server_pid" || stop_server "$server_pid"' EXIT

run_arm() {
    local rep=$1 arm=$2
    local label="r${rep}-${arm}"
    local server_log="$OUT/${label}-server.log"
    local load_log="$OUT/${label}-load.log"
    local -a policy

    case "$arm" in
        N) policy=(MEMRA_SERVE_SPEC=0) ;;
        K1) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1) ;;
        K2) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=2 MEMRA_SPEC_STATS=1) ;;
        K3) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=3 MEMRA_SPEC_STATS=1) ;;
        *) echo "FAIL: unknown arm $arm"; return 1 ;;
    esac

    if ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
        echo "FAIL: port ${PORT} occupied before $label"
        return 1
    fi

    echo "=== $label $(date -u +%FT%TZ) ==="
    env \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_GATE \
        -u MEMRA_SPEC_K \
        -u MEMRA_SPEC_STATS \
        -u MEMRA_SPEC_PP_ANATOMY \
        "${policy[@]}" \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_MODELS="step37=${MODEL}+${DRAFT}" \
        MEMRA_ADDR="$ADDR" \
        "$BIN" > "$server_log" 2>&1 &
    server_pid=$!

    if ! wait_up "$server_pid"; then
        echo "FAIL: $label server did not become ready"
        tail -100 "$server_log" || true
        return 1
    fi

    python3 tools/load-serve.py \
        --base "$BASE" \
        --model step37 \
        --concurrency 1 \
        --requests 4 \
        --max-tokens 128 \
        --greedy \
        --warmup 1 \
        --label "$label" \
        --out "$OUT/points.jsonl" \
        > "$load_log" 2>&1
    cat "$load_log"
    curl -sf "$BASE/metrics" > "$OUT/${label}-metrics.txt"
    gpu_sample
    sleep 1
    stop_server "$server_pid"
    server_pid=
    sleep 2

    local spec_lines
    spec_lines=$(grep -c '\[spec-acc\]' "$server_log" || true)
    if [[ "$arm" == N && "$spec_lines" -ne 0 ]]; then
        echo "FAIL: plain arm $label emitted $spec_lines spec lines"
        return 1
    fi
    if [[ "$arm" != N && "$spec_lines" -eq 0 ]]; then
        echo "FAIL: spec arm $label emitted no spec lines"
        return 1
    fi
    if grep -E -i 'illegal|sentinel|panic|abort|CUDA_ERROR|spec pending flush failed' \
        "$server_log" > "$OUT/${label}-error-scan.log"; then
        echo "FAIL: error signature in $label"
        cat "$OUT/${label}-error-scan.log"
        return 1
    fi
}

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "KSWEEP_LOCK_ACQUIRED $(date -u +%FT%TZ)"
echo "host=$(hostname) binary-source=7cd010c9 harness-source=lane-k-sweep-v1"
sha256sum "$BIN" > "$OUT/binary-sha256.txt"
stat -c '%n %s bytes' "$MODEL" "$DRAFT" > "$OUT/artifacts.txt"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-pre.csv" 2>&1 || true
gpu_sample

# Rotated/reversed ordering keeps every arm distributed through the thermal window.
for arm in N K1 K2 K3; do run_arm 1 "$arm"; done
for arm in K3 K2 K1 N; do run_arm 2 "$arm"; done
for arm in K1 N K3 K2; do run_arm 3 "$arm"; done
for arm in K2 K3 N K1; do run_arm 4 "$arm"; done
for arm in N K2 K3 K1; do run_arm 5 "$arm"; done

nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
    --format=csv,noheader > "$OUT/gpu-processes-post.csv" 2>&1 || true
gpu_sample
echo "KSWEEP_LOCK_RELEASED $(date -u +%FT%TZ)"
flock -u 9
echo "KSWEEP_DONE"
