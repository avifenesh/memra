#!/usr/bin/env bash
# memra#1258 (laptop, one gpulock lease on the RTX 5090): warm-pool margin sweep m0..m150 on a
# binary that prints the [diag1255] gate-pass line. usage: b-sweep.sh NAME BIN_ARM [extra args]
set -u
NAME=$1; ARM=$2; shift 2
S=$HOME/.cache/memra-1255
B=$HOME/.cache/memra-1258
M=$HOME/models/q38-gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$B/sweep/$NAME
mkdir -p "$OUT"
{ date -Is; sha256sum "$S/bin/memra-server-$ARM-120a"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader; echo "profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"; } > "$OUT/guard.log"
setsid python3 "$B/b-sweep.py" --bin "$S/bin/memra-server-$ARM-120a" --model "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
  --draft "$M/Qwen3.8-27B-DFlash2-memra" --out "$OUT" "$@" > "$OUT/sweep.log" 2>&1 &
P=$!
echo "$P" >> "$S/pids.txt"
trap 'kill -TERM -- -"$P" 2>/dev/null; exit 143' TERM INT
wait "$P"; rc=$?
kill -- -"$P" 2>/dev/null
echo "sweep $NAME rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
tail -n 20 "$OUT/sweep.log"
exit $rc
