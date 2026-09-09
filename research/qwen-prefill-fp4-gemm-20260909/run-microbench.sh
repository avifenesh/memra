#!/usr/bin/env bash
set -euo pipefail
lane_dir=/tmp/qwen-prefill-fp4-gemm-20260909
exec 9>/tmp/memra-gpu.lock
flock -w 45 9 || exit 75
export MEMRA_GPU_LOCK=/tmp/memra-gpu.lock MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=/root/target-fp4gemm RUSTC_WRAPPER=
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$apps" ]]; then printf 'REFUSED nonempty compute list: %s\n' "$apps"; exit 76; fi
nonce=$(cat /proc/sys/kernel/random/uuid)
printf '%s\n' "$nonce" > "$lane_dir/boot-nonce.txt"
date -u +%FT%TZ > "$lane_dir/start.txt"
nvidia-smi --query-gpu=name,uuid,temperature.gpu,clocks.sm,power.draw,power.limit --format=csv > "$lane_dir/preflight.csv"
printf '%s\n' "$$" > "$lane_dir/wrapper.pid"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used --format=csv -lms 250 > "$lane_dir/telemetry.csv" &
telemetry_pid=$!
printf '%s\n' "$telemetry_pid" > "$lane_dir/telemetry.pid"
cleanup() { kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; }
trap cleanup EXIT
/root/target-fp4gemm/microbench > "$lane_dir/microbench.jsonl" 2> "$lane_dir/microbench.stderr" &
bench_pid=$!
printf '%s\n' "$bench_pid" > "$lane_dir/bench.pid"
wait "$bench_pid"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$lane_dir/postflight.csv"
date -u +%FT%TZ > "$lane_dir/end.txt"
