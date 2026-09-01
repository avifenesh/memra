#!/usr/bin/env bash
# One-lock box1 performance battery for the increment-1 two-session spec pipeline.
set -euo pipefail

ROOT=${SPECMECH_ROOT:-/home/ubuntu/memra-specmech}
OUT=${SPECMECH_PERF_OUT:-/home/ubuntu/specmech-receipts/perf}
PORT=${SPECMECH_PERF_PORT:-8143}
BASE=http://127.0.0.1:${PORT}
MODEL=/home/ubuntu/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
DRAFT=/home/ubuntu/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf
SERVER=${ROOT}/target/release/memra-server
LOAD=${ROOT}/tools/load-serve.py

test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
cd "$ROOT"
exec > >(tee "$OUT/driver.log") 2>&1

for artifact in "$MODEL" "$DRAFT" "$SERVER" "$LOAD"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,temperature.gpu,pstate,clocks.sm,power.draw,power.limit \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } > "$path" 2>&1
}

wait_up() {
    local pid=$1 log=$2
    for _ in $(seq 1 450); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || { tail -100 "$log"; return 1; }
        sleep 2
    done
    tail -100 "$log"
    return 1
}

wait_idle() {
    for _ in $(seq 1 90); do
        test -z "$(compute_apps)" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

server_pid=
stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 90); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            wait_idle
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server $pid did not stop"
    return 1
}
trap stop_server EXIT INT TERM

assert_server_clean() {
    local log=$1 failures
    failures=$(grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|illegal|sentinel|spec pending flush failed' \
        "$log" || true)
    if [[ -n "$failures" ]]; then
        echo "$failures"
        echo "FAIL: server failure signature in $log"
        return 1
    fi
}

run_arm() {
    local shape=$1 rep=$2 arm=$3 trace=${4:-0}
    local label
    label=$(printf '%s-r%02d-%s' "$shape" "$rep" "$arm")
    if [[ "$trace" -eq 1 ]]; then
        label="${shape}-trace-${arm}"
    fi
    local log="$OUT/${label}-server.log"
    local -a policy trace_policy

    case "$arm" in
        plain) policy=(MEMRA_SERVE_SPEC=0) ;;
        serial) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1) ;;
        pipe) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1 MEMRA_SPEC_PIPE=1) ;;
        policy) policy=(MEMRA_SPEC_DEVACC=1 MEMRA_SPEC_PIPE=1) ;;
        *) echo "FAIL: unknown arm $arm"; return 1 ;;
    esac
    if [[ "$trace" -eq 1 ]]; then
        trace_policy=(MEMRA_SPEC_PHASE=1 MEMRA_TICK_TRACE=1)
    else
        trace_policy=()
    fi

    local concurrency requests warmup
    case "$shape" in
        c1) concurrency=1; requests=4; warmup=1 ;;
        c2) concurrency=2; requests=8; warmup=2 ;;
        c4) concurrency=4; requests=8; warmup=4 ;;
        *) echo "FAIL: unknown shape $shape"; return 1 ;;
    esac

    if ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
        echo "FAIL: port $PORT occupied before $label"
        return 1
    fi
    echo "=== $label $(date -u +%FT%TZ) ==="
    snapshot "$OUT/${label}-thermal-before.log" "$label-before"
    env \
        -u MEMRA_SERVE_SPEC \
        -u MEMRA_SPEC_GATE \
        -u MEMRA_SPEC_K \
        -u MEMRA_SPEC_STATS \
        -u MEMRA_SPEC_DEVACC \
        -u MEMRA_SPEC_PIPE \
        -u MEMRA_SPEC_PHASE \
        -u MEMRA_TICK_TRACE \
        -u MEMRA_SPEC_PP_ANATOMY \
        -u MEMRA_SPEC_REPLAY \
        -u MEMRA_SPEC_STREAM \
        "${policy[@]}" \
        "${trace_policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 \
        MEMRA_PREFILL_TICK=2048 \
        MEMRA_MODELS="step37=${MODEL}+${DRAFT}" \
        MEMRA_ADDR="127.0.0.1:${PORT}" \
        "$SERVER" > "$log" 2>&1 &
    server_pid=$!
    wait_up "$server_pid" "$log"

    python3 "$LOAD" \
        --base "$BASE" \
        --model step37 \
        --concurrency "$concurrency" \
        --requests "$requests" \
        --max-tokens 128 \
        --greedy \
        --warmup "$warmup" \
        --label "$label" \
        --out "$OUT/points.jsonl" \
        --per-request "$OUT/requests.jsonl" \
        > "$OUT/${label}-load.log" 2>&1
    cat "$OUT/${label}-load.log"
    curl -sf "$BASE/metrics" > "$OUT/${label}-metrics.txt"
    stop_server
    assert_server_clean "$log"
    snapshot "$OUT/${label}-thermal-after.log" "$label-after"

    local spec_lines pipe_lines
    spec_lines=$(grep -c '\[spec-acc\]' "$log" || true)
    pipe_lines=$(grep -c '\[spec-pipe\]' "$log" || true)
    echo "label=$label spec_lines=$spec_lines pipe_lines=$pipe_lines"
    case "$arm" in
        plain)
            test "$spec_lines" -eq 0
            test "$pipe_lines" -eq 0
            ;;
        serial)
            test "$spec_lines" -gt 0
            test "$pipe_lines" -eq 0
            ;;
        pipe)
            test "$spec_lines" -gt 0
            if [[ "$shape" == c2 ]]; then
                test "$pipe_lines" -gt 0
            else
                test "$pipe_lines" -eq 0
            fi
            ;;
        policy)
            test "$spec_lines" -eq 0
            test "$pipe_lines" -eq 0
            grep -q '\[spec-k\].*K=0.*source=pp2-placement' "$log"
            ;;
    esac
}

exec 9>/tmp/memra-gpu.lock
flock -w 900 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
echo "PERF_LOCK_ACQUIRED $(date -u +%FT%TZ)"
echo "host=$(hostname) source_commit=$(git rev-parse HEAD)"
git status --short --branch
sha256sum "$SERVER" "$LOAD" > "$OUT/SHA256SUMS"
stat -c '%n %s bytes %y' "$MODEL" "$DRAFT" > "$OUT/artifacts.txt"
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

# c=2 primary result: five interleaved observations per arm.
for arm in plain serial pipe; do run_arm c2 1 "$arm"; done
for arm in pipe plain serial; do run_arm c2 2 "$arm"; done
for arm in serial pipe plain; do run_arm c2 3 "$arm"; done
for arm in plain pipe serial; do run_arm c2 4 "$arm"; done
for arm in serial plain pipe; do run_arm c2 5 "$arm"; done

# One traced observation per speculative schedule for gap decomposition; excluded from N=5.
run_arm c2 0 serial 1
run_arm c2 0 pipe 1

# c=1: PIPE has no peer and must be a throughput-neutral serial fallback (N=5 interleaved).
for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then
        run_arm c1 "$rep" serial
        run_arm c1 "$rep" pipe
    else
        run_arm c1 "$rep" pipe
        run_arm c1 "$rep" serial
    fi
done

# c=4: default PP-2 policy must remain K=0 with the pipeline door open (N=5 interleaved).
for rep in 1 2 3 4 5; do
    if (( rep % 2 == 1 )); then
        run_arm c4 "$rep" plain
        run_arm c4 "$rep" policy
    else
        run_arm c4 "$rep" policy
        run_arm c4 "$rep" plain
    fi
done

snapshot "$OUT/nvidia-smi-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "PERF_PASS $(date -u +%FT%TZ)"
