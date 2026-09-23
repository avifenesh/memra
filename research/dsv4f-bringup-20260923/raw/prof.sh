#!/usr/bin/env bash
# prof.sh <label> <arm|current> [VAR=val ...]: one dsv4_decode_profile run under the box lock.
set -uo pipefail
label=$1; arm=$2; shift 2
out=/root/rcpt/prof/$label; mkdir -p "$out"
exec 9>/tmp/memra-gpu.lock
flock -n 9 || { echo SLOT_BUSY; exit 75; }
bin=/root/lane/target/release/dsv4_decode_profile
sha256sum "$bin" > "$out/binary.sha256"; printf '%s\n' "$@" > "$out/env.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm --format=csv -lms 250 > "$out/telemetry-250ms.csv" 2>&1 9>&- & tp=$!
a=(); [[ "$arm" != current ]] && a=("$arm")
export PATH=/usr/local/cuda/bin:$PATH
if [[ "${NSYS:-0}" == 1 ]]; then
  env "$@" nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx,osrt --cuda-graph-trace=node -o "$out/nsys" -f true "$bin" /data/dsv4f/nvfp4 /root/box/tape-9e3c8b550.txt "${a[@]}" > "$out/run.log" 2>&1
else
  env "$@" "$bin" /data/dsv4f/nvfp4 /root/box/tape-9e3c8b550.txt "${a[@]}" > "$out/run.log" 2>&1
fi
rc=$?; kill $tp
echo "PROF $label rc=$rc $(grep -E 'PROFILE_COMPLETE' "$out/run.log" | grep -oE 'seconds=[0-9.]+')" | tee -a /root/rcpt/prof/summary.txt
