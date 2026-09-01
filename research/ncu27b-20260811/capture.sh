#!/usr/bin/env bash
# Reproducible, bounded local-5090 profiler harness for the q27 speculative-decode lane.
# Every GPU-touching phase owns /tmp/memra-gpu.lock for exactly one process block.
set -euo pipefail

PHASE=${1:?usage: capture.sh <environment|nsys|export-nsys|ncu REGEX|export-ncu>}
KERNEL_REGEX=${2:-}

ROOT=$(git rev-parse --show-toplevel)
LANE=$ROOT/research/ncu27b-20260811
RAW=$LANE/raw
MODEL=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf
DRAFT=/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf
PROMPT=$ROOT/research/e2e/prompts/p1-code-short.txt
RUN_SPEC=$ROOT/target/release/run-spec
NSYS=/usr/local/cuda-13.1/bin/nsys
NCU=/usr/local/cuda-13.1/bin/ncu
NSYS_OUT=$RAW/spec-k3-n64
NCU_OUT=$RAW/ncu-spec-k3-n64
LOCK_WAIT=${MEMRA_LOCK_WAIT:-60}

mkdir -p "$RAW"

lock_gpu() {
    exec 9>/tmp/memra-gpu.lock
    flock -w "$LOCK_WAIT" 9 || {
        echo "GPU lock unavailable after ${LOCK_WAIT}s" >&2
        return 75
    }
}

unlock_gpu() {
    flock -u 9
    exec 9>&-
}

gpu_state() {
    echo "[gpu $(date -u +%FT%TZ)]"
    nvidia-smi \
        --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
        | sed 's/^/[compute-app] /'
}

run_contract() {
    echo "tree $(git -C "$ROOT" rev-parse HEAD)"
    echo "branch $(git -C "$ROOT" branch --show-current)"
    echo "run-spec-sha256 $(sha256sum "$RUN_SPEC" | cut -d' ' -f1)"
    echo "model-sha256 $(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "draft-sha256 $(sha256sum "$DRAFT" | cut -d' ' -f1)"
    echo "prompt-sha256 $(sha256sum "$PROMPT" | cut -d' ' -f1)"
    echo "shape c=1 K=3 NGEN=64 CHAT=1 PROFILE_SPEC=2 CUDA_VISIBLE_DEVICES=0"
}

profile_env=(
    CUDA_VISIBLE_DEVICES=0
    MEMRA_MTP_DRAFT="$DRAFT"
    MEMRA_SPEC_K=3
    MEMRA_NGEN=64
    MEMRA_PROMPT_FILE="$PROMPT"
    MEMRA_CHAT=1
    MEMRA_PROFILE_SPEC=2
    MEMRA_SPEC_PHASE=1
    MEMRA_SPEC_STATS=1
)

case "$PHASE" in
environment)
    LOG=$RAW/environment.log
    lock_gpu
    {
        echo "capture-start $(date -u +%FT%TZ)"
        run_contract
        "$NSYS" --version
        "$NCU" --version
        nvcc --version
        grep -E 'RmProfilingAdminOnly|EnableGpuFirmware' /proc/driver/nvidia/params || true
        printf 'platform-profile '; cat /sys/firmware/acpi/platform_profile
        gpu_state
        echo "capture-end $(date -u +%FT%TZ)"
    } 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    unlock_gpu
    exit "$rc"
    ;;
nsys)
    LOG=$RAW/nsys-spec-k3-n64.log
    lock_gpu
    set +e
    {
        echo "capture-start $(date -u +%FT%TZ)"
        run_contract
        gpu_state
        echo "profiler nsys; range=cudaProfilerApi; range-end=none; trace=cuda,nvtx; graph=node"
        timeout 2400 nice -n 5 env "${profile_env[@]}" \
            "$NSYS" profile \
            --output "$NSYS_OUT" \
            --force-overwrite=true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=none \
            --trace=cuda,nvtx \
            --cuda-graph-trace=node \
            --cuda-memory-usage=false \
            --sample=none \
            "$RUN_SPEC" "$MODEL"
        profile_rc=$?
        gpu_state
        echo "capture-end $(date -u +%FT%TZ) rc=$profile_rc"
        exit "$profile_rc"
    } 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    unlock_gpu
    exit "$rc"
    ;;
export-nsys)
    nice -n 10 "$NSYS" stats --report cuda_gpu_kern_sum --format csv \
        --output "$RAW/.nsys-spec-k3-n64-kern-sum" \
        "$NSYS_OUT.nsys-rep" > "$RAW/nsys-spec-k3-n64-kern-sum-export.log" 2>&1
    mv "$RAW/.nsys-spec-k3-n64-kern-sum_cuda_gpu_kern_sum.csv" \
        "$RAW/nsys-spec-k3-n64-kern-sum.csv"
    nice -n 10 "$NSYS" stats --report cuda_gpu_trace --format csv \
        --output "$RAW/.nsys-spec-k3-n64-trace" \
        "$NSYS_OUT.nsys-rep" > "$RAW/nsys-spec-k3-n64-trace-export.log" 2>&1
    mv "$RAW/.nsys-spec-k3-n64-trace_cuda_gpu_trace.csv" \
        "$RAW/nsys-spec-k3-n64-trace.csv"
    nice -n 10 "$NSYS" stats --report cuda_api_sum --format csv \
        --output "$RAW/.nsys-spec-k3-n64-api-sum" \
        "$NSYS_OUT.nsys-rep" > "$RAW/nsys-spec-k3-n64-api-sum-export.log" 2>&1
    mv "$RAW/.nsys-spec-k3-n64-api-sum_cuda_api_sum.csv" \
        "$RAW/nsys-spec-k3-n64-api-sum.csv"
    nice -n 10 "$NSYS" stats --report cuda_gpu_mem_time_sum --format csv \
        --output "$RAW/.nsys-spec-k3-n64-mem-sum" \
        "$NSYS_OUT.nsys-rep" > "$RAW/nsys-spec-k3-n64-mem-sum-export.log" 2>&1
    mv "$RAW/.nsys-spec-k3-n64-mem-sum_cuda_gpu_mem_time_sum.csv" \
        "$RAW/nsys-spec-k3-n64-mem-sum.csv"
    ;;
ncu)
    test -n "$KERNEL_REGEX" || {
        echo "ncu phase requires a selected-kernel regular expression" >&2
        exit 2
    }
    LOG=$RAW/ncu-spec-k3-n64.log
    printf '%s\n' "$KERNEL_REGEX" > "$RAW/ncu-kernel-regex.txt"
    lock_gpu
    set +e
    {
        echo "capture-start $(date -u +%FT%TZ)"
        run_contract
        gpu_state
        echo "profiler ncu; profile-from-start=off; graph=node; per-launch-config; one launch/config; base clocks"
        printf 'kernel-regex %s\n' "$KERNEL_REGEX"
        timeout 3600 nice -n 5 sudo -n env "${profile_env[@]}" \
            "$NCU" \
            --profile-from-start off \
            --graph-profiling node \
            --kernel-name-base function \
            --kernel-name "regex:$KERNEL_REGEX" \
            --filter-mode per-launch-config \
            --launch-count 1 \
            --clock-control base \
            --cache-control all \
            --section SpeedOfLight \
            --section MemoryWorkloadAnalysis \
            --section Occupancy \
            --section LaunchStats \
            --section SchedulerStats \
            --section WarpStateStats \
            --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
            --force-overwrite \
            --export "$NCU_OUT" \
            "$RUN_SPEC" "$MODEL"
        profile_rc=$?
        sudo -n chown "$(id -u):$(id -g)" "$NCU_OUT.ncu-rep" 2>/dev/null || true
        gpu_state
        echo "capture-end $(date -u +%FT%TZ) rc=$profile_rc"
        exit "$profile_rc"
    } 2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    unlock_gpu
    exit "$rc"
    ;;
export-ncu)
    "$NCU" --import "$NCU_OUT.ncu-rep" --csv --page raw \
        > "$RAW/ncu-spec-k3-n64-raw.csv" \
        2> "$RAW/ncu-spec-k3-n64-raw-export.log"
    "$NCU" --import "$NCU_OUT.ncu-rep" --page details \
        > "$RAW/ncu-spec-k3-n64-details.txt" \
        2> "$RAW/ncu-spec-k3-n64-details-export.log"
    ;;
*)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac
