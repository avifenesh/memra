#!/usr/bin/env bash
set -euo pipefail
name=${1:?cell binary name required}
case "$name" in int8-roof|int8-tiles|ablation-bench|exact-bench) ;; *) exit 2;; esac
lane=/tmp/qwen-prefill-fp4-gemm-20260909
out=$lane/$name
mkdir -p "$out"
exec 9>/tmp/memra-gpu.lock
flock -w 45 9 || exit 75
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$apps" ]]; then printf 'REFUSED nonempty compute list: %s\n' "$apps"; exit 76; fi
export RUSTC_WRAPPER= MEMRA_CUDA_ARCH=120a MEMRA_GPU_LOCK=/tmp/memra-gpu.lock CARGO_TARGET_DIR=/root/target-fp4gemm
unset MEMRA_PP_PIPE
nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,power.draw --format=csv > "$out/preflight.csv"
cat /proc/sys/kernel/random/uuid > "$out/nonce.txt"
date -u +%FT%TZ > "$out/start.txt"
nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,memory.used --format=csv -lms 250 > "$out/telemetry.csv" &
tpid=$!
trap 'kill "$tpid" 2>/dev/null || true; wait "$tpid" 2>/dev/null || true' EXIT
"$CARGO_TARGET_DIR/$name" > "$out/raw.jsonl" 2> "$out/stderr.log" &
bpid=$!
printf '%s %s %s\n' "$$" "$tpid" "$bpid" > "$out/pids.txt"
wait "$bpid"
date -u +%FT%TZ > "$out/end.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/postflight.csv"
