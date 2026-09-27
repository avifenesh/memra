#!/usr/bin/env bash
# DAY37 1.5 stage 0 on the local RTX 5090: the vmm-call-cost probe under the canonical lock,
# with 250 ms telemetry and compute-apps before and after. Usage: run-stage0.sh <out-dir> <binary>
set -euo pipefail
out=$1; bin=$2
mkdir -p "$out"
sha256sum "$bin" | tee "$out/binary.sha256"
git -C "$(dirname "$0")" rev-parse HEAD > "$out/source.txt"
exec 9>/tmp/memra-5090.lock
if ! flock -w 3600 9; then echo "lock not acquired within 3600 s; cell not run" | tee "$out/lock.txt"; exit 3; fi
date -u +%FT%TZ | sed 's/^/lock acquired /' | tee "$out/lock.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/compute-apps-before.csv"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv > "$out/gpu-before.csv"
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu --format=csv -lms 250 > "$out/samples.csv" &
smi=$!
set +e
"$bin" --reps 20 --queue-ms 60 --extents 1,4,16,64 2>&1 | tee "$out/probe.log"
rc=${PIPESTATUS[0]}
set -e
kill "$smi" 2>/dev/null || true
wait "$smi" 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$out/compute-apps-after.csv"
nvidia-smi --query-gpu=name,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used --format=csv > "$out/gpu-after.csv"
echo "exit=$rc" | tee "$out/exit.txt"
exit "$rc"
