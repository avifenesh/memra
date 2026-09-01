#!/usr/bin/env bash
# Bounded remote blocks for Qwen3.6-27B beside Step-3.7 on the Vast pair.
set -euo pipefail

BLOCK=${1:-}
case "$BLOCK" in
    A|B|C|final) ;;
    *) echo "usage: $0 A|B|C|final" >&2; exit 2 ;;
esac

REPO=/workspace/memra
BIN=$REPO/target/release/memra-server
MEASURE=/root/cx-27bab/measure.py
RECEIPTS=/root/cx-27bab/receipts
OUT=$RECEIPTS/$BLOCK
STEP_BASE=http://127.0.0.1:8002
Q27_BASE=http://127.0.0.1:8003
STEP_ALIAS=stepfun/step-3.7-flash
Q27_ALIAS=q27
Q27_ROOT=/workspace/models/qwen36-27b-nvfp4-mtp
Q27_MODEL=$Q27_ROOT/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q27_MTP=$Q27_ROOT/mtp-Qwen3.6-27B-NVFP4.gguf
Q27_DRAFT=$Q27_ROOT/draft-daily-owntrim-nvfp4head-q4blk.gguf

if [[ -e $OUT ]]; then
    echo "refusing to overwrite existing block receipt: $OUT" >&2
    exit 2
fi
mkdir -p "$OUT"

STEP_PID=
Q27_PID=
SOAK_PID=
GPU_PID=
KEEP_RUNNING=0
EXTRA_PIDS=()
LAST_BG_PID=

stop_pid() {
    local pid=${1:-}
    [[ -n $pid ]] || return 0
    kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 180); do
        kill -0 "$pid" 2>/dev/null || { wait "$pid" 2>/dev/null || true; return 0; }
        sleep 1
    done
    echo "graceful-stop-timeout pid=$pid"
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

finish() {
    local rc=$1
    trap - EXIT
    set +e
    if [[ $KEEP_RUNNING -eq 0 ]]; then
        for pid in "${EXTRA_PIDS[@]}"; do stop_pid "$pid"; done
        stop_pid "$SOAK_PID"
        stop_pid "$Q27_PID"
        stop_pid "$STEP_PID"
    fi
    stop_pid "$GPU_PID"
    printf '%s\n' "$rc" > "$OUT/runner.rc"
    printf 'block=%s exit=%s ended_utc=%s\n' "$BLOCK" "$rc" "$(date -u +%FT%TZ)"
    exit "$rc"
}
trap 'finish $?' EXIT

exec > >(tee "$OUT/driver.log") 2>&1

start_sampler() {
    nvidia-smi \
        --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,clocks.sm,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -l 1 > "$OUT/gpu.csv" 2>&1 &
    GPU_PID=$!
}

wait_up() {
    local base=$1 pid=$2
    for _ in $(seq 1 900); do
        curl -sf "$base/readyz" >/dev/null 2>&1 && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 2
    done
    return 1
}

wait_active() {
    local base=$1 required=$2 pid=$3
    for _ in $(seq 1 400); do
        kill -0 "$pid" 2>/dev/null || return 1
        local active
        active=$(python3 -c \
            'import json,sys,urllib.request; print(json.load(urllib.request.urlopen(sys.argv[1]+"/metrics", timeout=2)).get("active_sessions", 0))' \
            "$base" 2>/dev/null || echo 0)
        [[ $active -ge $required ]] && return 0
        sleep 0.05
    done
    return 1
}

kill_existing() {
    pkill -f '^python3 /root/soak.py$' 2>/dev/null || true
    for pid in $(pgrep -x memra-server || true); do
        if [[ $(readlink "/proc/$pid/cwd" 2>/dev/null || true) == "$REPO" ]]; then
            echo "stopping pre-existing memra-server pid=$pid"
            stop_pid "$pid"
        fi
    done
    for _ in $(seq 1 60); do
        if ! ss -ltn | grep -qE ':8002[[:space:]]|:8003[[:space:]]'; then return 0; fi
        sleep 1
    done
    ss -ltnp | grep -E ':8002|:8003' || true
    return 1
}

start_step() {
    local log=$1
    (
        set -a
        # Owner-supplied Step deployment template.  Reassert bounce because peer copies on
        # this exact host have a byte-integrity failure.
        source /root/serve-env.sh
        set +a
        export MEMRA_PP_HOST_BOUNCE=1
        export MEMRA_ADDR=0.0.0.0:8002
        exec "$BIN"
    ) > "$log" 2>&1 &
    STEP_PID=$!
    wait_up "$STEP_BASE" "$STEP_PID" || {
        echo "Step failed readiness pid=$STEP_PID"
        tail -160 "$log" || true
        return 1
    }
    echo "Step ready pid=$STEP_PID at $(date -u +%FT%TZ)"
    tr '\0' '\n' < "/proc/$STEP_PID/environ" \
        | grep -E '^(CUDA_VISIBLE_DEVICES|MEMRA_(ADDR|MODELS|PP_[A-Z0-9_]+|CTX|MOE_GROUPED|PREFILL_TICK|PREFIX_CACHE_MB))=' \
        | sort > "${log%.log}.env"
    grep -q '^MEMRA_PP_HOST_BOUNCE=1$' "${log%.log}.env"
    grep -q '^MEMRA_PP_STAGES=2$' "${log%.log}.env"
}

start_q27() {
    local log=$1
    (
        unset MEMRA_PP_STAGES MEMRA_PP_DEVICES MEMRA_PP_SHARD MEMRA_PP_HOST_BOUNCE
        unset MEMRA_MOE_GROUPED MEMRA_PREFILL_TICK MEMRA_SPEC_GATE
        unset MEMRA_SPEC_GATE_LOW MEMRA_SPEC_GATE_HIGH MEMRA_PREFIX_DEDUP
        export CUDA_VISIBLE_DEVICES=0
        export LD_LIBRARY_PATH=/usr/local/cuda-13.1/compat:/usr/local/cuda-13.1/lib64
        export MEMRA_ADDR=0.0.0.0:8003
        export MEMRA_COMPAT=openai
        export MEMRA_MODELS="$Q27_ALIAS=$Q27_MODEL+$Q27_DRAFT"
        export MEMRA_CTX=32768
        export MEMRA_PREFIX_CACHE_MB=0
        export MEMRA_SERVE_SPEC=1
        export MEMRA_SPEC_K=3
        exec "$BIN"
    ) > "$log" 2>&1 &
    Q27_PID=$!
    wait_up "$Q27_BASE" "$Q27_PID" || {
        echo "Q27 failed readiness pid=$Q27_PID"
        tail -160 "$log" || true
        return 1
    }
    echo "Q27 ready pid=$Q27_PID at $(date -u +%FT%TZ)"
    tr '\0' '\n' < "/proc/$Q27_PID/environ" \
        | grep -E '^(CUDA_VISIBLE_DEVICES|MEMRA_(ADDR|MODELS|PP_[A-Z0-9_]+|CTX|SERVE_SPEC|SPEC_K|PREFIX_CACHE_MB))=' \
        | sort > "${log%.log}.env"
    grep -q '^CUDA_VISIBLE_DEVICES=0$' "${log%.log}.env"
    grep -q '^MEMRA_CTX=32768$' "${log%.log}.env"
    grep -q '^MEMRA_SERVE_SPEC=1$' "${log%.log}.env"
}

run_measure() {
    local label=$1 base=$2 model=$3 prompt=$4 concurrency=$5 requests=$6 max_tokens=$7
    local expected=${8:-}
    local session_key=${9:-}
    local -a expected_arg=()
    local -a session_arg=()
    [[ -n $expected ]] && expected_arg=(--expected-sha256 "$expected")
    [[ -n $session_key ]] && session_arg=(--session-key "$session_key")
    echo "--- measure $label at $(date -u +%FT%TZ)"
    python3 "$MEASURE" \
        --base "$base" --model "$model" --label "$label" --prompt "$prompt" \
        --concurrency "$concurrency" --requests "$requests" --max-tokens "$max_tokens" \
        --rows "$OUT/$label.requests.jsonl" --summary "$OUT/$label.summary.json" \
        "${expected_arg[@]}" \
        "${session_arg[@]}" \
        2>&1 | tee "$OUT/$label.console.log"
}

start_duration_load() {
    local label=$1 base=$2 model=$3 prompt=$4 concurrency=$5 duration=$6 max_tokens=$7
    echo "--- duration-load $label at $(date -u +%FT%TZ)"
    python3 "$MEASURE" \
        --base "$base" --model "$model" --label "$label" --prompt "$prompt" \
        --concurrency "$concurrency" --requests 1 --duration "$duration" \
        --max-tokens "$max_tokens" \
        --session-key cx27-q27-background \
        --rows "$OUT/$label.requests.jsonl" --summary "$OUT/$label.summary.json" \
        > "$OUT/$label.console.log" 2>&1 &
    LAST_BG_PID=$!
    EXTRA_PIDS+=("$LAST_BG_PID")
}

extract_hash() {
    python3 -c \
        'import json,sys; d=json.load(open(sys.argv[1])); h=d["text_hash_counts"]; assert len(h)==1 and d["bos_garbage_count"]==0 and d["n_error"]==0, d; print(next(iter(h)))' \
        "$1"
}

check_long_class() {
    python3 -c \
        'import json,sys; d=json.load(open(sys.argv[1])); v=d["prompt_tokens_observed"]; assert len(v)==1 and 3800 <= v[0] <= 4400, v; print("long_prompt_tokens="+str(v[0]))' \
        "$1"
}

check_overlap() {
    local left=$1 right=$2
    python3 - "$left" "$right" <<'PY'
import json
import pathlib
import sys

def interval(path):
    rows = [json.loads(line) for line in pathlib.Path(path).read_text().splitlines()]
    rows = [row for row in rows if row.get("ok")]
    assert rows, f"no successful rows in {path}"
    return min(row["started_monotonic_s"] for row in rows), max(
        row["ended_monotonic_s"] for row in rows
    )

left = interval(sys.argv[1])
right = interval(sys.argv[2])
overlap = min(left[1], right[1]) - max(left[0], right[0])
assert overlap > 0, {"left": left, "right": right, "overlap_s": overlap}
print(f"overlap_s={overlap:.6f} left={sys.argv[1]} right={sys.argv[2]}")
PY
}

snapshot() {
    local name=$1
    {
        printf 'captured_utc=%s\n' "$(date -u +%FT%TZ)"
        printf '%s\n' '[gpu]'
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,pstate,power.draw,clocks.sm \
            --format=csv,noheader
        printf '%s\n' '[compute-apps]'
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
        printf '%s\n' '[processes]'
        ps -eo pid,lstart,cmd | grep -E 'memra-server|soak.py' | grep -v grep || true
        printf '%s\n' '[ports]'
        ss -ltnp | grep -E ':8002|:8003' || true
        printf '%s\n' '[step-metrics]'
        curl -sS "$STEP_BASE/metrics" || true
        printf '\n%s\n' '[q27-metrics]'
        curl -sS "$Q27_BASE/metrics" || true
        printf '\n'
    } > "$OUT/$name.txt" 2>&1
}

scan_failure_log() {
    local log=$1 out=$2
    grep -nE 'CUDA_ERROR|out of memory|panicked at|segmentation fault|illegal address|FATAL' \
        "$log" > "$out" || true
    if [[ -s $out ]]; then
        echo "failure signature found in $log"
        cat "$out"
        return 1
    fi
}

provenance() {
    {
        printf 'block=%s\nstarted_utc=%s\nhost=%s\n' "$BLOCK" "$(date -u +%FT%TZ)" "$(hostname)"
        cd "$REPO"
        git status --short --branch
        git show -s --format='commit=%H%nsubject=%s' HEAD
        sha256sum "$BIN" "$MEASURE" /root/serve-env.sh
        stat -c '%n %s bytes %y' "$Q27_MODEL" "$Q27_MTP" "$Q27_DRAFT"
        /usr/local/cuda-13.1/bin/nvcc --version | tail -1
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
    } > "$OUT/provenance.txt" 2>&1
}

run_step_probe_set() {
    local suffix=$1 expected=$2
    run_measure "step-short-$suffix" "$STEP_BASE" "$STEP_ALIAS" sanity 1 1 32 "$expected"
    run_measure "step-decode-$suffix" "$STEP_BASE" "$STEP_ALIAS" workload 1 1 256
}

sanity_both() {
    local suffix=$1 step_expected=$2 q27_expected=$3
    run_measure "sanity-step-$suffix" "$STEP_BASE" "$STEP_ALIAS" sanity 1 1 32 "$step_expected" cx27-step-sanity
    run_measure "sanity-q27-$suffix" "$Q27_BASE" "$Q27_ALIAS" sanity 1 1 32 "$q27_expected" cx27-q27-sanity
}

run_step_active_set() {
    local rep=$1 expected=$2
    local bg_pid
    start_duration_load "q27-bg-r$rep" "$Q27_BASE" "$Q27_ALIAS" workload 2 14 128
    bg_pid=$LAST_BG_PID
    # The client has two continuously replenished workers. The single-card scheduler can
    # publish active_sessions=1 between batches even while both client intervals overlap,
    # so require one observed in-flight request only as the timed-overlap start barrier.
    wait_active "$Q27_BASE" 1 "$bg_pid" || {
        echo "q27 c=2 background failed to establish live overlap"
        tail -120 "$OUT/q27-bg-r$rep.console.log" || true
        return 1
    }
    run_step_probe_set "active-r$rep" "$expected"
    set +e
    wait "$bg_pid"
    local rc=$?
    set -e
    cat "$OUT/q27-bg-r$rep.console.log"
    [[ $rc -eq 0 ]]
    check_overlap "$OUT/q27-bg-r$rep.requests.jsonl" "$OUT/step-short-active-r$rep.requests.jsonl"
    check_overlap "$OUT/q27-bg-r$rep.requests.jsonl" "$OUT/step-decode-active-r$rep.requests.jsonl"
}

run_q27_prime_probe() {
    local rep=$1
    local prime="step-prime-r$rep"
    python3 "$MEASURE" \
        --base "$STEP_BASE" --model "$STEP_ALIAS" --label "$prime" --prompt long4k \
        --concurrency 1 --requests 1 --max-tokens 8 \
        --rows "$OUT/$prime.requests.jsonl" --summary "$OUT/$prime.summary.json" \
        > "$OUT/$prime.console.log" 2>&1 &
    local prime_pid=$!
    EXTRA_PIDS+=("$prime_pid")
    sleep 0.25
    kill -0 "$prime_pid" 2>/dev/null || {
        echo "Step 4k prime exited before the Q27 overlap probe"
        cat "$OUT/$prime.console.log" || true
        return 1
    }
    run_measure "q27-under-step-prime-r$rep" "$Q27_BASE" "$Q27_ALIAS" workload 1 1 256
    set +e
    wait "$prime_pid"
    local rc=$?
    set -e
    cat "$OUT/$prime.console.log"
    [[ $rc -eq 0 ]]
    check_long_class "$OUT/$prime.summary.json"
    check_overlap "$OUT/$prime.requests.jsonl" "$OUT/q27-under-step-prime-r$rep.requests.jsonl"
}

echo "=== cx-27bab block $BLOCK start $(date -u +%FT%TZ) ==="
kill_existing
start_sampler
provenance

case "$BLOCK" in
    A)
        start_step "$OUT/step-server.log"
        step_hash=
        for rep in $(seq 1 5); do
            if ((rep % 2 == 1)); then order='short long decode c4'; else order='c4 decode long short'; fi
            for cell in $order; do
                case "$cell" in
                    short)
                        run_measure "step-short-r$rep" "$STEP_BASE" "$STEP_ALIAS" sanity 1 1 32 "$step_hash"
                        if [[ -z $step_hash ]]; then
                            step_hash=$(extract_hash "$OUT/step-short-r$rep.summary.json")
                            printf '%s\n' "$step_hash" > "$OUT/step-known.sha256"
                        fi
                        ;;
                    long)
                        run_measure "step-4k-r$rep" "$STEP_BASE" "$STEP_ALIAS" long4k 1 1 8
                        check_long_class "$OUT/step-4k-r$rep.summary.json"
                        ;;
                    decode)
                        run_measure "step-decode-c1-r$rep" "$STEP_BASE" "$STEP_ALIAS" workload 1 1 256
                        ;;
                    c4)
                        run_measure "step-decode-c4-r$rep" "$STEP_BASE" "$STEP_ALIAS" workload 4 4 256
                        ;;
                esac
            done
        done
        snapshot final
        stop_pid "$STEP_PID"
        STEP_PID=
        scan_failure_log "$OUT/step-server.log" "$OUT/step-server.failures.txt"
        ;;
    B)
        start_q27 "$OUT/q27-server.log"
        q27_hash=
        for rep in $(seq 1 5); do
            if ((rep % 2 == 1)); then order='short decode c4'; else order='c4 decode short'; fi
            for cell in $order; do
                case "$cell" in
                    short)
                        run_measure "q27-short-r$rep" "$Q27_BASE" "$Q27_ALIAS" sanity 1 1 32 "$q27_hash"
                        if [[ -z $q27_hash ]]; then
                            q27_hash=$(extract_hash "$OUT/q27-short-r$rep.summary.json")
                            printf '%s\n' "$q27_hash" > "$OUT/q27-known.sha256"
                        fi
                        ;;
                    decode)
                        run_measure "q27-decode-c1-r$rep" "$Q27_BASE" "$Q27_ALIAS" workload 1 1 256
                        ;;
                    c4)
                        run_measure "q27-decode-c4-r$rep" "$Q27_BASE" "$Q27_ALIAS" workload 4 4 256
                        ;;
                esac
            done
        done
        snapshot final
        stop_pid "$Q27_PID"
        Q27_PID=
        scan_failure_log "$OUT/q27-server.log" "$OUT/q27-server.failures.txt"
        ;;
    C)
        [[ -s $RECEIPTS/A/step-known.sha256 && -s $RECEIPTS/B/q27-known.sha256 ]]
        step_hash=$(tr -d '\n' < "$RECEIPTS/A/step-known.sha256")
        q27_hash=$(tr -d '\n' < "$RECEIPTS/B/q27-known.sha256")
        start_step "$OUT/step-server.log"
        start_q27 "$OUT/q27-server.log"
        sanity_both initial "$step_hash" "$q27_hash"
        snapshot vram-both-resident-before

        # Pair an idle co-resident control with the requested steady c=2 Q27 load.  Order
        # alternates per rep so the active-vs-idle contention delta is interleaved.
        for rep in $(seq 1 5); do
            if ((rep % 2 == 1)); then order='idle active'; else order='active idle'; fi
            for arm in $order; do
                case "$arm" in
                    idle) run_step_probe_set "idle-r$rep" "$step_hash" ;;
                    active) run_step_active_set "$rep" "$step_hash" ;;
                esac
                sanity_both "forward-$arm-post-r$rep" "$step_hash" "$q27_hash"
            done
        done

        sanity_both before-reverse "$step_hash" "$q27_hash"

        # Reverse interference: pair Q27's both-resident idle control with a request whose
        # complete lifetime overlaps a fresh Step ~4k prime.
        for rep in $(seq 1 5); do
            if ((rep % 2 == 1)); then order='idle prime'; else order='prime idle'; fi
            for arm in $order; do
                case "$arm" in
                    idle)
                        run_measure "q27-step-idle-r$rep" "$Q27_BASE" "$Q27_ALIAS" workload 1 1 256
                        ;;
                    prime)
                        run_q27_prime_probe "$rep"
                        ;;
                esac
                sanity_both "reverse-$arm-post-r$rep" "$step_hash" "$q27_hash"
            done
        done

        sanity_both final "$step_hash" "$q27_hash"
        snapshot vram-both-resident-after
        stop_pid "$Q27_PID"
        Q27_PID=
        stop_pid "$STEP_PID"
        STEP_PID=
        scan_failure_log "$OUT/step-server.log" "$OUT/step-server.failures.txt"
        scan_failure_log "$OUT/q27-server.log" "$OUT/q27-server.failures.txt"
        ;;
    final)
        [[ -s $RECEIPTS/A/step-known.sha256 && -s $RECEIPTS/B/q27-known.sha256 ]]
        step_hash=$(tr -d '\n' < "$RECEIPTS/A/step-known.sha256")
        q27_hash=$(tr -d '\n' < "$RECEIPTS/B/q27-known.sha256")
        start_step /var/log/memra-step-27bab.log
        start_q27 /var/log/memra-q27-27bab.log
        run_measure final-sanity-step "$STEP_BASE" "$STEP_ALIAS" sanity 1 1 32 "$step_hash"
        run_measure final-sanity-q27 "$Q27_BASE" "$Q27_ALIAS" sanity 1 1 32 "$q27_hash"
        pkill -f '^python3 /root/soak.py$' 2>/dev/null || true
        nohup python3 /root/soak.py > /var/log/soak-driver.log 2>&1 < /dev/null &
        SOAK_PID=$!
        sleep 5
        kill -0 "$SOAK_PID"
        snapshot running-final
        tail -200 /var/log/memra-step-27bab.log > "$OUT/step-server-tail.log"
        tail -200 /var/log/memra-q27-27bab.log > "$OUT/q27-server-tail.log"
        tail -20 /var/log/soak.jsonl > "$OUT/soak-tail.jsonl" 2>/dev/null || true
        scan_failure_log /var/log/memra-step-27bab.log "$OUT/step-server.failures.txt"
        scan_failure_log /var/log/memra-q27-27bab.log "$OUT/q27-server.failures.txt"
        KEEP_RUNNING=1
        ;;
esac

echo "=== cx-27bab block $BLOCK PASS $(date -u +%FT%TZ) ==="
