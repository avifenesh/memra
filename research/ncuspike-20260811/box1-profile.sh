#!/usr/bin/env bash
# Detached, lock-held box1 profiler driver for the resident Step-3.7 decode path.
set -euo pipefail

PHASE=${1:?usage: box1-profile.sh <nsys|ncu> [kernel-regex]}
KERNEL_REGEX=${2:-}

REPO=${NCUSPIKE_REPO:-/opt/dl-image/nvme/memra-cx-ncuspike-src}
TARGET=${NCUSPIKE_TARGET:-/opt/dl-image/nvme/memra-cx-ncuspike-target}
RUN_ROOT=${NCUSPIKE_RUN_ROOT:-/opt/dl-image/nvme/ncuspike-20260811}
MODEL_ROOT=${NCUSPIKE_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf
PROMPT=$REPO/tools/fast-gate/prompts/probe.txt
RUN_GEN=$TARGET/release/run-gen
PROFILE_BIN=${NCUSPIKE_PROFILE_BIN:-$RUN_ROOT/profile-target/release/ncuspike-profile}
NSYS=/usr/local/bin/nsys
NCU=/usr/local/cuda-13.2/bin/ncu
EXPECTED_SHA=1808220ead39d515a0854df49d1bb6452b558209
LOCK_WAIT=${NCUSPIKE_LOCK_WAIT:-28800}
NCU_DEVICES=${NCUSPIKE_NCU_DEVICES:-0,1}
NCU_OUT=${NCUSPIKE_NCU_OUT:-ncu}
NSYS_OUT=${NCUSPIKE_NSYS_OUT:-nsys}

compute_apps() {
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        df -h /tmp /opt/dl-image/nvme
        free -h
        swapon --show
        nvidia-smi \
            --query-gpu=index,name,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

acquire_idle_lock() {
    local start now apps
    start=$(date +%s)
    exec 9>/tmp/memra-gpu.lock
    while true; do
        if flock -n 9; then
            apps=$(compute_apps || true)
            if [[ -z $apps ]]; then
                echo "GPU_LOCK_ACQUIRED ts=$(date -u +%FT%TZ)"
                return 0
            fi
            echo "GPU_IDLE_GATE_BUSY ts=$(date -u +%FT%TZ)"
            printf '%s\n' "$apps"
            flock -u 9
        else
            echo "GPU_LOCK_BUSY ts=$(date -u +%FT%TZ)"
        fi
        now=$(date +%s)
        if (( now - start >= LOCK_WAIT )); then
            echo "FAIL: no idle lock window within ${LOCK_WAIT}s"
            return 75
        fi
        sleep 15
    done
}

release_lock() {
    flock -u 9 2>/dev/null || true
    exec 9>&- 2>/dev/null || true
}

host_pressure_gate() {
    local tmp_avail_kib mem_avail_kib swap_total_kib swap_free_kib swap_used_kib
    tmp_avail_kib=$(df -Pk /tmp | awk 'NR == 2 {print $4}')
    mem_avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    swap_total_kib=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
    swap_free_kib=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
    swap_used_kib=$((swap_total_kib - swap_free_kib))
    echo "host_pressure tmp_avail_kib=$tmp_avail_kib mem_avail_kib=$mem_avail_kib swap_used_kib=$swap_used_kib swap_total_kib=$swap_total_kib"
    (( tmp_avail_kib >= 32 * 1024 * 1024 )) || {
        echo "FAIL: host-pressure red: /tmp has less than 32 GiB available"
        return 1
    }
    (( mem_avail_kib >= 64 * 1024 * 1024 )) || {
        echo "FAIL: host-pressure red: MemAvailable is below 64 GiB"
        return 1
    }
    (( swap_total_kib == 0 || swap_used_kib * 100 < swap_total_kib * 80 )) || {
        echo "FAIL: host-pressure red: swap is at least 80% used"
        return 1
    }
    echo "HOST_PRESSURE_PASS"
}

contract() {
    echo "host=$(hostname)"
    echo "source_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "source_status=$(git -C "$REPO" status --porcelain | wc -l)"
    echo "run_gen_sha256=$(sha256sum "$RUN_GEN" | awk '{print $1}')"
    echo "profile_bin=$PROFILE_BIN"
    echo "profile_bin_sha256=$(sha256sum "$PROFILE_BIN" | awk '{print $1}')"
    echo "model_first_part=$MODEL"
    find "$MODEL_ROOT/IQ4_XS" -maxdepth 1 -type f -name '*.gguf' \
        -printf 'model_part=%f bytes=%s mtime=%TY-%Tm-%TdT%TH:%TM:%TSZ\n' | sort
    echo "model_total_bytes=$(find "$MODEL_ROOT/IQ4_XS" -maxdepth 1 -type f -name '*.gguf' -printf '%s\n' | awk '{s += $1} END {printf "%.0f", s}')"
    echo "model_receipt=/home/ubuntu/memra-cx-sigrouter2/research/throughput-20260810/raw/block-baseline-20260809T222030Z/artifact-sha256.txt"
    sed -n '1,3p' /home/ubuntu/memra-cx-sigrouter2/research/throughput-20260810/raw/block-baseline-20260809T222030Z/artifact-sha256.txt
    echo "shape=PP2 devices=0,1 stage-owned-KV resident-default sigmoid-router serving B=1 synthetic-depth=512"
    echo "clock_regime=stock 600W no application clock or power cap"
    "$NSYS" --version
    "$NCU" --version
}

profile_env=(
    "CUDA_VISIBLE_DEVICES=0,1"
    "MEMRA_PP_STAGES=2"
    "MEMRA_PP_DEVICES=0,1"
    "MEMRA_CTX=262144"
    "MEMRA_MOE_GROUPED=1"
    "MEMRA_SERVE_SPEC=0"
)

test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$REPO" status --porcelain)"
for artifact in "$MODEL" "$PROMPT" "$RUN_GEN" "$PROFILE_BIN" "$NSYS" "$NCU"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
mkdir -p "$RUN_ROOT"

case "$PHASE" in
nsys)
    OUT=$RUN_ROOT/$NSYS_OUT
    test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
    mkdir -p "$OUT"
    exec > >(tee "$OUT/driver.log") 2>&1
    echo "phase=nsys start=$(date -u +%FT%TZ)"
    contract | tee "$OUT/contract.txt"
    acquire_idle_lock
    snapshot "$OUT/host-before.txt" lock-acquired
    host_pressure_gate
    set +e
    timeout 3600 nice -n 10 env \
        -u MEMRA_SIG_ROUTER -u MEMRA_MOE_DEV -u MEMRA_SERVE_BATCH \
        -u MEMRA_SPEC_K -u MEMRA_BG_JOB -u MEMRA_SERVE_B1FAST \
        "${profile_env[@]}" \
        "$NSYS" profile \
        --output "$OUT/decode" \
        --force-overwrite=true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --trace=cuda,nvtx \
        --cuda-graph-trace=node \
        --cuda-memory-usage=false \
        --sample=none \
        --cpuctxsw=none \
        "$PROFILE_BIN" "$MODEL" 512 32 2>&1 | tee "$OUT/nsys-run.log"
    rc=${PIPESTATUS[0]}
    set -e
    snapshot "$OUT/host-after.txt" profiler-exit
    release_lock
    echo "nsys_rc=$rc"
    (( rc == 0 )) || exit "$rc"
    rep=$OUT/decode.nsys-rep
    sha256sum "$rep" >"$OUT/report.sha256"
    "$NSYS" stats --report cuda_gpu_kern_sum "$rep" \
        >"$OUT/cuda-gpu-kern-sum.txt" 2>"$OUT/cuda-gpu-kern-sum.stderr"
    "$NSYS" stats --report cuda_gpu_kern_sum --format csv "$rep" \
        >"$OUT/cuda-gpu-kern-sum.csv" 2>"$OUT/cuda-gpu-kern-sum-csv.stderr"
    "$NSYS" stats --report cuda_gpu_trace --format csv "$rep" \
        >"$OUT/cuda-gpu-trace.csv" 2>"$OUT/cuda-gpu-trace.stderr"
    "$NSYS" stats --report cuda_api_sum "$rep" \
        >"$OUT/cuda-api-sum.txt" 2>"$OUT/cuda-api-sum.stderr"
    "$NSYS" stats --report cuda_gpu_mem_time_sum "$rep" \
        >"$OUT/cuda-gpu-mem-time-sum.txt" 2>"$OUT/cuda-gpu-mem-time-sum.stderr"
    rm -f "$rep" "$OUT/decode.sqlite"
    echo "phase=nsys done=$(date -u +%FT%TZ)"
    ;;
ncu)
    test -n "$KERNEL_REGEX" || { echo "FAIL: ncu requires a kernel regex"; exit 2; }
    OUT=$RUN_ROOT/$NCU_OUT
    test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
    mkdir -p "$OUT"
    exec > >(tee "$OUT/driver.log") 2>&1
    echo "phase=ncu start=$(date -u +%FT%TZ)"
    echo "ncu_devices=$NCU_DEVICES"
    contract | tee "$OUT/contract.txt"
    printf '%s\n' "$KERNEL_REGEX" >"$OUT/kernel-regex.txt"
    acquire_idle_lock
    snapshot "$OUT/host-before.txt" lock-acquired
    host_pressure_gate
    set +e
    timeout 7200 nice -n 10 sudo -n env \
        -u MEMRA_SIG_ROUTER -u MEMRA_MOE_DEV -u MEMRA_SERVE_BATCH \
        -u MEMRA_SPEC_K -u MEMRA_BG_JOB -u MEMRA_SERVE_B1FAST \
        "${profile_env[@]}" \
        "$NCU" \
        --profile-from-start off \
        --graph-profiling node \
        --devices "$NCU_DEVICES" \
        --kernel-name-base function \
        --kernel-name "regex:$KERNEL_REGEX" \
        --filter-mode per-launch-config \
        --launch-count 1 \
        --clock-control none \
        --pipeline-boost-state dynamic \
        --cache-control all \
        --metrics gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.per_second,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,launch__registers_per_thread,launch__waves_per_multiprocessor,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_wait_per_issue_active.ratio,smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio \
        --force-overwrite \
        --export "$OUT/decode" \
        "$PROFILE_BIN" "$MODEL" 512 2 2>&1 | tee "$OUT/ncu-run.log"
    rc=${PIPESTATUS[0]}
    set -e
    sudo -n chown "$(id -u):$(id -g)" "$OUT/decode.ncu-rep" 2>/dev/null || true
    snapshot "$OUT/host-after.txt" profiler-exit
    release_lock
    echo "ncu_rc=$rc"
    (( rc == 0 )) || exit "$rc"
    rep=$OUT/decode.ncu-rep
    sha256sum "$rep" >"$OUT/report.sha256"
    "$NCU" --import "$rep" --csv --page raw \
        >"$OUT/raw.csv" 2>"$OUT/raw-export.stderr"
    "$NCU" --import "$rep" --page details \
        >"$OUT/details.txt" 2>"$OUT/details-export.stderr"
    rm -f "$rep"
    echo "phase=ncu done=$(date -u +%FT%TZ)"
    ;;
*)
    echo "FAIL: unknown phase $PHASE"
    exit 2
    ;;
esac
