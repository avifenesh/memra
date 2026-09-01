#!/usr/bin/env bash
# Build or run the single-card cx-cachesize campaign on box1.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
    build|run|run-resume|diagnose-seed|diagnose-restore) ;;
    *) echo "usage: $0 build|run|run-resume|diagnose-seed|diagnose-restore" >&2; exit 2 ;;
esac

export PATH=/home/ubuntu/.cargo/bin:/usr/local/cuda-13.2/bin:$PATH
ROOT=${CACHESIZE_ROOT:-/opt/dl-image/nvme/cx-cachesize}
REPO=${CACHESIZE_REPO:-$ROOT/memra}
HARNESS=${CACHESIZE_HARNESS:-$ROOT/harness}
MODELS=${CACHESIZE_MODELS:-/opt/dl-image/nvme/cx-requal/models}
EXPECTED_SOURCE=18885ec479d897a3e8c42b0d408a71fa3edaa708
EXPECTED_Q27=d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517
EXPECTED_Q35=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
EXPECTED_REPLAY=91eac7250e0d268ac6be8cfd1ee64e346d405dc412824dab45f224e9563e1e5b
EXPECTED_WORKLOAD=85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34

Q27=$MODELS/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
Q35=$MODELS/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
SERVER=$REPO/target/release/memra-server
FROZEN_REPLAY=$REPO/research/sellgate-20260812/sellgate_replay.py
WORKLOAD_LOCK=$REPO/research/sellgate-20260812/workload.lock.json
PROTOCOL=$HARNESS/protocol.lock.json
ENTRY_PROBE=$HARNESS/measure_entry_bytes.py
SWEEP=$HARNESS/capacity_sweep.py
RESTORE_ORACLE=$HARNESS/restore_oracle.py
STAMP=${CACHESIZE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${CACHESIZE_OUT:-$ROOT/raw/scored-$STAMP}
LOCK_WAIT_SECONDS=${CACHESIZE_LOCK_WAIT_SECONDS:-0}

SERVER_PID=
BOOT_SAMPLER_PID=
GLOBAL_SAMPLER_PID=
ALL_GPU_SAMPLER_PID=
VMSTAT_PID=
DMON_PID=

compute_apps() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null || true
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,\
power.draw,power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,\
utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
        compute_apps | sed 's/^/[compute-app] /'
    } >"$path" 2>&1
}

start_all_gpu_sampler() {
    (
        while true; do
            date -u +%FT%T.%3NZ
            nvidia-smi --query-gpu=index,name,uuid,temperature.gpu,power.draw,\
memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
            sleep 0.25
        done
    ) >"$OUT/operator-gpu-both-250ms.csv" 2>&1 &
    ALL_GPU_SAMPLER_PID=$!
}

source_preflight() {
    test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SOURCE"
    test "$(git -C "$REPO" describe --tags --exact-match HEAD)" = v0.81.2
    local dirty
    dirty=$(git -C "$REPO" status --porcelain --untracked-files=all)
    test -z "$dirty" || { echo "$dirty"; echo "FAIL: runtime checkout is dirty"; return 1; }
    test "$(sha256sum "$Q27" | awk '{print $1}')" = "$EXPECTED_Q27"
    test "$(sha256sum "$Q35" | awk '{print $1}')" = "$EXPECTED_Q35"
    test "$(sha256sum "$FROZEN_REPLAY" | awk '{print $1}')" = "$EXPECTED_REPLAY"
    test "$(sha256sum "$WORKLOAD_LOCK" | awk '{print $1}')" = "$EXPECTED_WORKLOAD"
    python3 -m json.tool "$PROTOCOL" >/dev/null
    python3 -m py_compile "$ENTRY_PROBE" "$SWEEP" "$RESTORE_ORACLE"
    bash -n "$HARNESS/run-box1.sh"
}

stop_boot_sampler() {
    test -n "${BOOT_SAMPLER_PID:-}" || return 0
    kill "$BOOT_SAMPLER_PID" 2>/dev/null || true
    wait "$BOOT_SAMPLER_PID" 2>/dev/null || true
    BOOT_SAMPLER_PID=
}

stop_server() {
    test -n "${SERVER_PID:-}" || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=
            stop_boot_sampler
            return 0
        fi
        sleep 1
    done
    echo "FAIL: owned server pid=$SERVER_PID did not stop after 120 seconds"
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
    stop_boot_sampler
    return 1
}

stop_global_samplers() {
    local pid
    for pid in "${GLOBAL_SAMPLER_PID:-}" "${ALL_GPU_SAMPLER_PID:-}" \
        "${VMSTAT_PID:-}" "${DMON_PID:-}"; do
        test -n "$pid" || continue
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    GLOBAL_SAMPLER_PID=
    ALL_GPU_SAMPLER_PID=
    VMSTAT_PID=
    DMON_PID=
}

cleanup() {
    stop_server || true
    stop_global_samplers
}

wait_ready() {
    local base=$1 log=$2
    for _ in $(seq 1 900); do
        curl -sf "$base/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "FAIL: server died during boot"
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    echo "FAIL: server readiness timeout"
    tail -200 "$log"
    return 1
}

start_server() {
    local model=$1 budget=$2 ctx=$3 run_dir=$4 port path
    case "$model" in
        q27) port=18427; path=$Q27 ;;
        q35) port=18435; path=$Q35 ;;
        *) return 2 ;;
    esac
    if ss -tln 2>/dev/null | grep -q "[:.]$port "; then
        echo "FAIL: port $port already has a listener"
        return 1
    fi
    snapshot "$run_dir/gpu-before.log" "$model-b${budget}-before"
    nvidia-smi -i 0 --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,\
power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$run_dir/gpu-250ms.csv" 2>&1 &
    BOOT_SAMPLER_PID=$!
    env -u MEMRA_PP_STAGES -u MEMRA_PP_DEVICES -u MEMRA_DUAL_PP \
        -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE -u MEMRA_PRIME_PIPE \
        -u MEMRA_PRIME_BATCH -u MEMRA_PREFILL_TICK -u MEMRA_SERVE_BATCH \
        -u MEMRA_SPEC_K -u MEMRA_SPEC_GATE -u MEMRA_DECODE_BATCH_CAP \
        -u MEMRA_FAST -u MEMRA_MOE_RESIDENT -u MEMRA_MOE_RESIDENT_GB \
        CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS="$model=$path" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$port" \
        MEMRA_TAG="cx-cachesize-$model-b$budget" MEMRA_SERVE_SPEC=0 \
        MEMRA_CTX="$ctx" MEMRA_PREFIX_CACHE_MB="$budget" MEMRA_PREFIX_DEDUP=1 \
        MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0 MEMRA_MAX_SESSIONS=96 \
        "$SERVER" >"$run_dir/server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready "http://127.0.0.1:$port" "$run_dir/server.log"
    curl -sf "http://127.0.0.1:$port/v1/models" >"$run_dir/models.json"
    curl -sf "http://127.0.0.1:$port/metrics" >"$run_dir/metrics-ready.json"
    snapshot "$run_dir/gpu-ready.log" "$model-b${budget}-ready"
}

run_entry_probe() {
    local model=$1 port run_dir rc
    case "$model" in
        q27) port=18427 ;;
        q35) port=18435 ;;
        *) return 2 ;;
    esac
    run_dir=$OUT/entry-bytes/$model
    mkdir -p "$run_dir"
    echo "ENTRY_PROBE_START model=$model ts=$(date -u +%FT%TZ)"
    start_server "$model" 2048 16384 "$run_dir"
    set +e
    timeout 7200 python3 "$ENTRY_PROBE" \
        --endpoint "http://127.0.0.1:$port" --model "$model" \
        --protocol "$PROTOCOL" --frozen-replay "$FROZEN_REPLAY" \
        --workload-lock "$WORKLOAD_LOCK" --out "$run_dir/entry-bytes.jsonl" \
        --namespace "cx-cachesize-entry" --timeout 1800 \
        2>&1 | tee "$run_dir/entry-bytes.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$run_dir/entry-bytes.exit"
    curl -sf "http://127.0.0.1:$port/metrics" >"$run_dir/metrics-final.json"
    snapshot "$run_dir/gpu-final.log" "$model-entry-final"
    stop_server
    test "$rc" -eq 0
    grep -q '"verdict": "PASS"' "$run_dir/entry-bytes.jsonl"
    test -z "$(compute_apps)" || { compute_apps; echo "FAIL: GPU process remained"; return 1; }
    echo "ENTRY_PROBE_PASS model=$model ts=$(date -u +%FT%TZ)"
}

run_budget_boot() {
    local model=$1 budget=$2 rep=$3 port run_dir rc
    case "$model" in
        q27) port=18427 ;;
        q35) port=18435 ;;
        *) return 2 ;;
    esac
    run_dir=$OUT/campaign/r$(printf '%02d' "$rep")-$model-b$(printf '%05d' "$budget")
    mkdir -p "$run_dir"
    echo "BUDGET_BOOT_START model=$model budget_mb=$budget rep=$rep ts=$(date -u +%FT%TZ)"
    test -z "$(compute_apps)" || {
        compute_apps
        echo "FAIL: foreign compute app present before budget boot"
        return 1
    }
    start_server "$model" "$budget" 8192 "$run_dir"
    set +e
    timeout 28800 python3 "$SWEEP" \
        --endpoint "http://127.0.0.1:$port" --model "$model" \
        --budget-mb "$budget" --repetition "$rep" --protocol "$PROTOCOL" \
        --frozen-replay "$FROZEN_REPLAY" --workload-lock "$WORKLOAD_LOCK" \
        --out "$run_dir/sweep.jsonl" \
        --namespace "cx-cachesize-$model-b$(printf '%05d' "$budget")-r$(printf '%02d' "$rep")" \
        --expected-server-pid "$SERVER_PID" \
        --timeout 1800 \
        2>&1 | tee "$run_dir/sweep.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$run_dir/sweep.exit"
    curl -sf "http://127.0.0.1:$port/metrics" >"$run_dir/metrics-final.json"
    snapshot "$run_dir/gpu-final.log" "$model-b${budget}-r${rep}-final"
    stop_server
    grep -Ein \
        'panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|MISMATCH' \
        "$run_dir/server.log" >"$run_dir/server-failure-scan.log" || true
    test ! -s "$run_dir/server-failure-scan.log"
    test "$rc" -eq 0
    grep -q '"verdict": "PASS"' "$run_dir/sweep.jsonl"
    test -z "$(compute_apps)" || { compute_apps; echo "FAIL: GPU process remained"; return 1; }
    echo "BUDGET_BOOT_PASS model=$model budget_mb=$budget rep=$rep ts=$(date -u +%FT%TZ)"
}

write_manifest() {
    local temp
    temp=$(mktemp "$OUT/.manifest.XXXXXX")
    (
        cd "$OUT"
        find . -type f ! -name MANIFEST.sha256 ! -name orchestrator.log \
            ! -name '.manifest.*' -print0 | sort -z | xargs -0 sha256sum
    ) >"$temp"
    mv "$temp" "$OUT/MANIFEST.sha256"
}

build_runtime() {
    source_preflight
    test ! -e "$OUT" || { echo "FAIL: output exists: $OUT"; return 1; }
    mkdir -p "$OUT"
    exec > >(tee "$OUT/build.log") 2>&1
    echo "BUILD_START ts=$(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
    cd "$REPO"
    nice -n 10 ionice -c 2 -n 7 cargo build --release -p memra-server --bin memra-server
    sha256sum "$SERVER" >"$OUT/runtime-binaries.sha256"
    git status --porcelain --untracked-files=all >"$OUT/git-status.txt"
    test ! -s "$OUT/git-status.txt"
    touch "$OUT/build.ok"
    echo "BUILD_PASS ts=$(date -u +%FT%TZ)"
}

run_campaign() {
    local resume=0
    if [[ "$MODE" == run-resume ]]; then
        resume=1
    fi
    test ! -e "$OUT" || { echo "FAIL: output exists: $OUT"; return 1; }
    mkdir -p "$OUT/campaign"
    if (( ! resume )); then
        mkdir -p "$OUT/entry-bytes"
    fi
    exec > >(tee "$OUT/orchestrator.log") 2>&1
    trap cleanup EXIT INT TERM
    echo "CACHESIZE_START ts=$(date -u +%FT%TZ) source=$EXPECTED_SOURCE pid=$$ resume=$resume"
    echo "LOCK_QUEUE_CHECK ts=$(date -u +%FT%TZ)"
    fuser -v /tmp/memra-gpu.lock 2>&1 || true
    fuser -v /tmp/memra-gpu-1.lock 2>&1 || true
    exec 9>/tmp/memra-gpu.lock
    flock -w "$LOCK_WAIT_SECONDS" 9 || {
        echo "FAIL: /tmp/memra-gpu.lock wait timed out after ${LOCK_WAIT_SECONDS}s"
        return 75
    }
    exec 8>/tmp/memra-gpu-1.lock
    flock -w "$LOCK_WAIT_SECONDS" 8 || {
        flock -u 9
        exec 9>&-
        echo "FAIL: /tmp/memra-gpu-1.lock wait timed out after ${LOCK_WAIT_SECONDS}s"
        return 75
    }
    echo "CACHESIZE_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"

    source_preflight
    test -x "$SERVER"
    sha256sum "$Q27" "$Q35" "$SERVER" "$PROTOCOL" "$ENTRY_PROBE" "$SWEEP" \
        "$FROZEN_REPLAY" "$WORKLOAD_LOCK" >"$OUT/SHA256SUMS.input"
    {
        echo "timestamp=$(date -u +%FT%TZ)"
        echo "runtime_source=$EXPECTED_SOURCE"
        echo "runtime_tag=v0.81.2"
        echo "segment_kind=$([[ $resume -eq 1 ]] && echo resume-57-boots || echo full-60-boots)"
        echo "shape=physical GPU0 only; GPU1 idle; one model resident at a time"
        echo "working_set_entries=96"
        echo "budgets_mb=1024,4096,8192,16384,32768,49152"
        echo "repetitions=5"
        echo "odd_repetitions=q27,q35 and ascending budgets"
        echo "even_repetitions=q35,q27 and descending budgets"
        hostname
        uname -a
        git -C "$REPO" log -5 --oneline --decorate
        rustc --version
        cargo --version
        nvcc --version
        nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
            --format=csv,noheader
    } >"$OUT/provenance.txt" 2>&1

    snapshot "$OUT/gpu-before.log" campaign-before
    test -z "$(compute_apps)" || { compute_apps; echo "FAIL: box1 GPUs are not idle"; return 1; }
    test "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -eq 2

    nvidia-smi -i 0 --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,\
power.limit,clocks.sm,clocks.mem,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits -lms 250 >"$OUT/gpu-250ms.csv" 2>&1 &
    GLOBAL_SAMPLER_PID=$!
    start_all_gpu_sampler
    vmstat 1 >"$OUT/vmstat-1s.log" 2>&1 &
    VMSTAT_PID=$!
    nvidia-smi dmon -i 0 -s pucmt -d 1 -o DT >"$OUT/dmon-1s.log" 2>&1 &
    DMON_PID=$!

    if (( resume )); then
        echo "ENTRY_PROBES_REUSED source=attempt7-cache-hit-eos"
        echo "BUDGET_BOOT_REUSED model=q27 budget_mb=1024 rep=1 source=attempt7-cache-hit-eos"
        echo "BUDGET_BOOT_REUSED model=q27 budget_mb=4096 rep=1 source=attempt7-cache-hit-eos"
        echo "BUDGET_BOOT_EXCLUDED model=q27 budget_mb=8192 rep=1 owner=cx-eosclass"
    else
        run_entry_probe q27
        run_entry_probe q35
    fi

    local rep model budget
    local -a models budgets
    for rep in 1 2 3 4 5; do
        if (( rep % 2 == 1 )); then
            models=(q27 q35)
            budgets=(1024 4096 8192 16384 32768 49152)
        else
            models=(q35 q27)
            budgets=(49152 32768 16384 8192 4096 1024)
        fi
        for model in "${models[@]}"; do
            for budget in "${budgets[@]}"; do
                if (( resume && rep == 1 )) && [[ "$model" == q27 ]] \
                    && (( budget == 1024 || budget == 4096 || budget == 8192 )); then
                    continue
                fi
                run_budget_boot "$model" "$budget" "$rep"
            done
        done
    done

    stop_global_samplers
    snapshot "$OUT/gpu-after.log" campaign-after
    {
        echo "ts=$(date -u +%FT%TZ)"
        echo "compute_apps:"
        compute_apps
        echo "memra_processes:"
        pgrep -af '[m]emra-server' || true
        echo "ports:"
        ss -tlnp | grep -E ':(18427|18435)\b' || true
        echo "gpu_memory_mib:"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
        echo "lock_holder:"
        fuser -v /tmp/memra-gpu.lock 2>&1 || true
        echo "gpu1_lock_holder:"
        fuser -v /tmp/memra-gpu-1.lock 2>&1 || true
    } >"$OUT/pre-unlock-clean.log" 2>&1
    test -z "$(compute_apps)"
    test -z "$(pgrep -af '[m]emra-server' || true)"
    test -z "$(ss -tlnp | grep -E ':(18427|18435)\b' || true)"
    test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -u)" = 0
    if (( resume )); then
        touch "$OUT/resume-segment.complete"
    else
        touch "$OUT/campaign.complete"
    fi
    write_manifest
    flock -u 8
    exec 8>&-
    flock -u 9
    exec 9>&-
    {
        echo "ts=$(date -u +%FT%TZ)"
        echo "lock_holder:"
        fuser -v /tmp/memra-gpu.lock 2>&1 || true
        echo "gpu1_lock_holder:"
        fuser -v /tmp/memra-gpu-1.lock 2>&1 || true
        echo "compute_apps:"
        compute_apps
        echo "ports:"
        ss -tlnp | grep -E ':(18427|18435)\b' || true
    } >"$OUT/post-unlock-clean.log" 2>&1
    trap - EXIT INT TERM
    echo "CACHESIZE_COMPLETE ts=$(date -u +%FT%TZ) out=$OUT resume=$resume"
}

run_seed_diagnostic() {
    source_preflight
    test -x "$SERVER"
    test ! -e "$OUT" || { echo "FAIL: output exists: $OUT"; return 1; }
    mkdir -p "$OUT/campaign"
    exec > >(tee "$OUT/orchestrator.log") 2>&1
    trap cleanup EXIT INT TERM
    echo "SEED_DIAGNOSTIC_START ts=$(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; return 75; }
    echo "SEED_DIAGNOSTIC_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
    test -z "$(compute_apps)" || { compute_apps; echo "FAIL: box1 GPUs are not idle"; return 1; }
    run_budget_boot q27 16384 1
    test -z "$(compute_apps)"
    test -z "$(pgrep -af '[m]emra-server' || true)"
    test -z "$(ss -tlnp | grep -E ':(18427|18435)\b' || true)"
    test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -u)" = 0
    touch "$OUT/diagnostic.complete"
    write_manifest
    flock -u 9
    exec 9>&-
    trap - EXIT INT TERM
    echo "SEED_DIAGNOSTIC_COMPLETE ts=$(date -u +%FT%TZ) out=$OUT"
}

run_restore_diagnostic() {
    source_preflight
    test -x "$SERVER"
    test ! -e "$OUT" || { echo "FAIL: output exists: $OUT"; return 1; }
    mkdir -p "$OUT/restore-oracle"
    exec > >(tee "$OUT/orchestrator.log") 2>&1
    trap cleanup EXIT INT TERM
    echo "RESTORE_DIAGNOSTIC_START ts=$(date -u +%FT%TZ) source=$EXPECTED_SOURCE"
    exec 9>/tmp/memra-gpu.lock
    flock -w 14400 9 || { echo "FAIL: GPU lock timeout"; return 75; }
    exec 8>/tmp/memra-gpu-1.lock
    flock -w 14400 8 || {
        flock -u 9
        exec 9>&-
        echo "FAIL: GPU1 coordination lock timeout"
        return 75
    }
    echo "RESTORE_DIAGNOSTIC_LOCK_ACQUIRED ts=$(date -u +%FT%TZ) pid=$$"
    test -z "$(compute_apps)" || { compute_apps; echo "FAIL: box1 GPUs are not idle"; return 1; }
    sha256sum "$Q27" "$SERVER" "$PROTOCOL" "$RESTORE_ORACLE" \
        "$FROZEN_REPLAY" "$WORKLOAD_LOCK" >"$OUT/SHA256SUMS.input"
    start_server q27 8192 8192 "$OUT/restore-oracle"
    set +e
    timeout 7200 python3 "$RESTORE_ORACLE" \
        --endpoint http://127.0.0.1:18427 --protocol "$PROTOCOL" \
        --frozen-replay "$FROZEN_REPLAY" --workload-lock "$WORKLOAD_LOCK" \
        --out "$OUT/restore-oracle/oracle.jsonl" \
        --namespace cx-cachesize-restore-oracle --budget-mb 8192 \
        --target-prefix-id 87 --repetitions 3 --timeout 1800 \
        2>&1 | tee "$OUT/restore-oracle/oracle.log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "$rc" >"$OUT/restore-oracle/oracle.exit"
    curl -sf http://127.0.0.1:18427/metrics >"$OUT/restore-oracle/metrics-final.json"
    snapshot "$OUT/restore-oracle/gpu-final.log" q27-restore-oracle-final
    stop_server
    grep -Ein \
        'panicked at|worker.*died|server.*FATAL|illegal memory access|ILLEGAL_ADDRESS|MISMATCH' \
        "$OUT/restore-oracle/server.log" >"$OUT/restore-oracle/server-failure-scan.log" || true
    test ! -s "$OUT/restore-oracle/server-failure-scan.log"
    test "$rc" -eq 0
    grep -q '"verdict": "PASS"' "$OUT/restore-oracle/oracle.jsonl"
    test -z "$(compute_apps)"
    test -z "$(pgrep -af '[m]emra-server' || true)"
    test -z "$(ss -tlnp | grep -E ':(18427|18435)\b' || true)"
    test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ' | sort -u)" = 0
    touch "$OUT/diagnostic.complete"
    write_manifest
    flock -u 8
    exec 8>&-
    flock -u 9
    exec 9>&-
    trap - EXIT INT TERM
    echo "RESTORE_DIAGNOSTIC_COMPLETE ts=$(date -u +%FT%TZ) out=$OUT"
}

case "$MODE" in
    build) build_runtime ;;
    run|run-resume) run_campaign ;;
    diagnose-seed) run_seed_diagnostic ;;
    diagnose-restore) run_restore_diagnostic ;;
esac
