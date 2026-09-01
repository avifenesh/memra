#!/usr/bin/env bash
# One-lock box1 c=2, N=5 interleaved floor + OPTIPIPE increment-2 q sweep.
set -euo pipefail

ROOT=${OPTI2_ROOT:-/home/ubuntu/memra-opti2}
OUT=${OPTI2_PERF_OUT:-/home/ubuntu/opti2-receipts/perf-c2-1}
PORT=${OPTI2_PERF_PORT:-8159}
LOCK_WAIT=${OPTI2_LOCK_WAIT:-900}
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

record_controller_telemetry() {
    local label=$1 arm=$2 threshold=$3 log=$4
    python3 - "$label" "$arm" "$threshold" "$log" >> "$OUT/controller-telemetry.jsonl" <<'PY'
import json
import re
import statistics
import sys

label, arm, threshold, path = sys.argv[1:]
text = open(path, encoding="utf-8", errors="replace").read()
issues = len(re.findall(r"\[opti-controller\] issue ", text))
rejects = len(re.findall(r"\[opti-controller\] reject ", text))
hits = len(re.findall(r"\[opti-controller\] resolve .* hit=true ", text))
misses = len(re.findall(r"\[opti-controller\] resolve .* hit=false ", text))
reconciles = len(re.findall(r"\[opti-controller\] resolve .* reconcile=true ", text))
tail_drains = len(re.findall(r"\[opti-controller\] tail-drain ", text))
breaker_trips = len(re.findall(r"\[opti-controller\] resolve .* breaker=true", text))
shadow_hits = len(re.findall(r"\[opti-controller\] shadow .* v_n=true ", text))
shadow_misses = len(re.findall(r"\[opti-controller\] shadow .* v_n=false ", text))
resolution_ms = [float(v) for v in re.findall(r"resolution_ms=([0-9.]+)", text)]
checks = issues + rejects
resolved = hits + misses
labels = resolved + shadow_hits + shadow_misses
row = {
    "label": label,
    "arm": arm,
    "threshold": float(threshold),
    "checks": checks,
    "admits": issues,
    "rejects": rejects,
    "hits": hits,
    "misses": misses,
    "reconciles": reconciles,
    "tail_drains": tail_drains,
    "breaker_trips": breaker_trips,
    "shadow_reject_hits": shadow_hits,
    "shadow_reject_misses": shadow_misses,
    "opportunity_labels": labels,
    "admitted_hit_rate": hits / resolved if resolved else None,
    "opportunity_hit_rate": (hits + shadow_hits) / labels if labels else None,
    "wasted_draft_tokens": rejects + 2 * misses + 2 * tail_drains,
    "shadow_draft_tokens": rejects + 2 * issues,
    "resolution_ms_median": statistics.median(resolution_ms) if resolution_ms else None,
    "resolution_ms_mean": statistics.mean(resolution_ms) if resolution_ms else None,
}
if checks == 0 or checks != labels or issues != resolved or misses != reconciles:
    raise SystemExit(f"controller accounting failed for {label}: {row}")
print(json.dumps(row, sort_keys=True))
PY
}

run_arm() {
    local rep=$1 arm=$2 trace=${3:-0}
    local label threshold=-1
    label=$(printf 'c2-r%02d-%s' "$rep" "$arm")
    if [[ "$trace" -eq 1 ]]; then
        label="c2-trace-${arm}"
    fi
    local log="$OUT/${label}-server.log"
    local -a policy trace_policy

    case "$arm" in
        plain)  policy=(MEMRA_SERVE_SPEC=0) ;;
        serial) policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1) ;;
        seam)   policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1 MEMRA_SPEC_PIPE=1) ;;
        q0)     threshold=0.0; policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1 MEMRA_OPTI_CONTROLLER_Q=0.0) ;;
        q05)    threshold=0.5; policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1 MEMRA_OPTI_CONTROLLER_Q=0.5) ;;
        q07)    threshold=0.7; policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1 MEMRA_OPTI_CONTROLLER_Q=0.7) ;;
        q09)    threshold=0.9; policy=(MEMRA_SPEC_GATE=0 MEMRA_SPEC_K=1 MEMRA_SPEC_STATS=1 MEMRA_SPEC_DEVACC=1 MEMRA_OPTI_CONTROLLER_Q=0.9) ;;
        *) echo "FAIL: unknown arm $arm"; return 1 ;;
    esac
    if [[ "$trace" -eq 1 ]]; then
        trace_policy=(MEMRA_SPEC_PHASE=1 MEMRA_SPEC_PP_ANATOMY=1)
    else
        trace_policy=()
    fi

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
        -u MEMRA_SPEC_PIPE_TRACE \
        -u MEMRA_SPEC_PHASE \
        -u MEMRA_SPEC_PP_ANATOMY \
        -u MEMRA_TICK_TRACE \
        -u MEMRA_OPTI_CONTROLLER_Q \
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
        --concurrency 2 \
        --requests 8 \
        --max-tokens 128 \
        --greedy \
        --warmup 2 \
        --label "$label" \
        --out "$OUT/points.jsonl" \
        --per-request "$OUT/requests.jsonl" \
        > "$OUT/${label}-load.log" 2>&1
    sed -n '1,120p' "$OUT/${label}-load.log"
    python3 - "$OUT/points.jsonl" "$label" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
row = rows[-1]
if row["label"] != sys.argv[2] or row["n_ok"] != 8 or row["n_err"] != 0 \
        or row["n_shed"] != 0 or row["completion_tokens_total"] != 1024:
    raise SystemExit(f"load point failed: {row}")
PY
    curl -sf "$BASE/metrics" > "$OUT/${label}-metrics.txt"
    stop_server
    assert_server_clean "$log"
    snapshot "$OUT/${label}-thermal-after.log" "$label-after"

    local spec_lines pipe_lines opti_lines
    spec_lines=$(grep -c '\[spec-acc\]' "$log" || true)
    pipe_lines=$(grep -c '\[spec-pipe\]' "$log" || true)
    opti_lines=$(grep -c '\[opti-controller\]' "$log" || true)
    echo "label=$label spec_lines=$spec_lines pipe_lines=$pipe_lines opti_lines=$opti_lines"
    case "$arm" in
        plain)  test "$spec_lines" -eq 0; test "$pipe_lines" -eq 0; test "$opti_lines" -eq 0 ;;
        serial) test "$spec_lines" -gt 0; test "$pipe_lines" -eq 0; test "$opti_lines" -eq 0 ;;
        seam)   test "$spec_lines" -gt 0; test "$pipe_lines" -gt 0; test "$opti_lines" -eq 0 ;;
        q*)
            test "$spec_lines" -gt 0
            test "$pipe_lines" -eq 0
            grep -q '\[opti-fork\] armed mode=Controller' "$log"
            record_controller_telemetry "$label" "$arm" "$threshold" "$log"
            ;;
    esac
    if [[ "$trace" -eq 1 ]]; then
        grep -q '\[spec-phase\]' "$log"
        grep -q '\[spec-anatomy\]' "$log"
    fi
}

exec 9>/tmp/memra-gpu.lock
flock -w "$LOCK_WAIT" 9 || { echo "FAIL: GPU lock timeout after ${LOCK_WAIT}s"; exit 75; }
echo "PERF_LOCK_ACQUIRED $(date -u +%FT%TZ)"
echo "host=$(hostname) source_commit=$(git rev-parse HEAD)"
git status --short --branch
sha256sum "$SERVER" "$LOAD" > "$OUT/SHA256SUMS"
stat -c '%n %s bytes %y' "$MODEL" "$DRAFT" > "$OUT/artifacts.txt"
snapshot "$OUT/nvidia-smi-before.log" lock-acquired
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

# Fixed before execution: cyclic order with a stride of three to spread each arm over the
# thermal progression. Every arm has N=5 under this single lock hold.
for arm in plain serial seam q0 q05 q07 q09; do run_arm 1 "$arm"; done
for arm in q0 q05 q07 q09 plain serial seam; do run_arm 2 "$arm"; done
for arm in q09 plain serial seam q0 q05 q07; do run_arm 3 "$arm"; done
for arm in seam q0 q05 q07 q09 plain serial; do run_arm 4 "$arm"; done
for arm in q07 q09 plain serial seam q0 q05; do run_arm 5 "$arm"; done

# Instrumented observations are excluded from the N=5 medians.
run_arm 0 q0 1
run_arm 0 q07 1

snapshot "$OUT/nvidia-smi-after.log" complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "PERF_PASS $(date -u +%FT%TZ)"
