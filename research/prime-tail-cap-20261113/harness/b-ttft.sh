#!/usr/bin/env bash
# memra#1258 (laptop, one gpulock lease on the RTX 5090): TTFT and prime walls of cold raw-text
# prompts of 4,090 / 4,100 / 4,106 / 4,111 tokens (b-ttft.py), boots interleaved over the arms
# given (memra-server-<arm>-120a). usage: b-ttft.sh ARM...   e.g. b-ttft.sh 1258base 1258 1258 1258base
set -u
S=$HOME/.cache/memra-1255
B=$HOME/.cache/memra-1258
M=$HOME/models/q38-gguf
P=$B/ttft-prompts
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$B/ttft/$(date +%H%M%S)
mkdir -p "$OUT"
{ date -Is; for a in "$@"; do sha256sum "$S/bin/memra-server-$a-120a"; done | sort -u
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader; } > "$OUT/guard.log"
rc=0
i=0
for arm in "$@"; do
  i=$((i + 1))
  setsid python3 "$B/b-ttft.py" --bin "$S/bin/memra-server-$arm-120a" --model "$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" \
    --draft "$M/Qwen3.8-27B-DFlash2-memra" --out "$OUT/b$i-$arm" --reps 3 \
    --prompts "$P/p4090.txt,$P/p4100.txt,$P/p4106.txt,$P/p4111.txt" > "$OUT/b$i-$arm.log" 2>&1 &
  PID=$!
  echo "$PID" >> "$S/pids.txt"
  trap 'kill -TERM -- -"$PID" 2>/dev/null; exit 143' TERM INT
  wait "$PID" || rc=$?
  kill -- -"$PID" 2>/dev/null
  echo "boot $i $arm done rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,enforced.power.limit --format=csv,noheader >> "$OUT/guard.log"
done
exit $rc
