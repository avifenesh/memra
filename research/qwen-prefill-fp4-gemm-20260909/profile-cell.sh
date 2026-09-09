#!/usr/bin/env bash
set -euo pipefail
shape=${1:?shape}
case "$shape" in 0|1) ;; *) exit 2;; esac
lane=/tmp/qwen-prefill-fp4-gemm-20260909
out=$lane/ncu-$shape
mkdir -p "$out"
exec 9>/tmp/memra-gpu.lock
flock -w 45 9 || exit 75
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$apps" ]]; then printf 'REFUSED compute list %s\n' "$apps";exit 76;fi
export MEMRA_CUDA_ARCH=120a RUSTC_WRAPPER= CARGO_TARGET_DIR=/root/target-fp4gemm
unset MEMRA_PP_PIPE
printf '%s\n' "$$" > "$out/wrapper.pid"
date -u +%FT%TZ > "$out/start.txt"
nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv > "$out/preflight.csv"
ncu --clock-control none --replay-mode kernel --kernel-name-base function --kernel-name regex:mul_mat_q_nvfp4_w4a8 --launch-count 1 \
 --section SpeedOfLight --section ComputeWorkloadAnalysis --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis --section Occupancy --section InstructionStats --section SourceCounters \
 --export "$out/profile" /root/target-fp4gemm/int8-profile "$shape" > "$out/console.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$out/ncu.pid"
wait "$pid"
ncu --import "$out/profile.ncu-rep" --page raw --csv > "$out/raw.csv"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/postflight.csv"
date -u +%FT%TZ > "$out/end.txt"
