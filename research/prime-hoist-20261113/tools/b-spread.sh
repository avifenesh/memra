#!/usr/bin/env bash
# memra#1258 (laptop, one gpulock lease on the RTX 5090): the first prime chunk's pool draw beside a
# decoding keeper, 20 cold 5.7k-token primes per boot (b-spread.py), boots interleaved over the
# MEMRA_DIAG_CALL_TRIM modes given (0 = main's behaviour, 1 = shortfall trim, 2 = drain per call).
# usage: b-spread.sh ARM MODE...
set -u
ARM=$1; shift
S=$HOME/.cache/memra-1255
B=$HOME/.cache/memra-1258
M=$HOME/models/q38-gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$B/spread/$ARM-$(date +%H%M%S)
mkdir -p "$OUT"
{ date -Is; sha256sum "$S/bin/memra-server-$ARM-120a"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader; echo "profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"; } > "$OUT/guard.log"
rc=0
i=0
for mode in "$@"; do
  i=$((i + 1))
  setsid python3 "$B/b-spread.py" --bin "$S/bin/memra-server-$ARM-120a" --model "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
    --draft "$M/Qwen3.8-27B-DFlash2-memra" --out "$OUT/b$i-mode$mode" --mode "$mode" ${SPREAD_ARGS:-} > "$OUT/b$i-mode$mode.log" 2>&1 &
  P=$!
  echo "$P" >> "$S/pids.txt"
  trap 'kill -TERM -- -"$P" 2>/dev/null; exit 143' TERM INT
  wait "$P" || rc=$?
  kill -- -"$P" 2>/dev/null
  echo "boot $i mode $mode done rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
done
exit $rc
