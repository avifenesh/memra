#!/usr/bin/env bash
# Rig, under the caller's gpulock lease: one cell of round 5 lever 1's RTX 5090 B/T. A fresh
# memra-server (Qwen3.8-27B NVFP4+Q5_K, the frspec 32k MTP draft, 32k context, timestamped log with
# MEMRA_TICK_TRACE=1), then eos_ab.py (g1k, g1k-t2, s1k), then stop.
# usage: eos5090_cell.sh CAMPAIGN_DIR NAME BIN_DIR
set -u
CD=$1; NAME=$2; BIN=$3
D=~/.cache/memra-gb10-r4
M=~/models/q38-gguf
OUT=$CD/$NAME; mkdir -p "$OUT"
export MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8095 MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=4 MEMRA_KV_HOST_MB=0
export MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_MTP_DRAFT="$M/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
export MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_TICK_TRACE=1
{ date '+%F %T'; sha256sum "$BIN/memra-server"; env | grep '^MEMRA_' | sort; } > "$OUT/guard.log"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
  --format=csv,noheader -lms 250 > "$OUT/gpu.csv" 2>&1 &
smi=$!
/usr/bin/python3 "$D/box/tsrun.py" "$BIN/memra-server" > "$OUT/server.log" 2>&1 &
sp=$!
if ! waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8095/readyz && exit 0 || exit 1' --fail-if "! kill -0 $sp" --every 5s --timeout 10m > "$OUT/run.log" 2>&1; then
  kill -TERM $sp; wait $sp; kill $smi; echo "EOS-CELL-FAIL $NAME" >> "$CD/campaign.log"; exit 1
fi
BENCH_BASE=http://127.0.0.1:8095 /usr/bin/python3 "$D/box/eos_ab.py" "$OUT/ab.jsonl" --tag "$NAME" >> "$OUT/run.log" 2>&1
kill -TERM $sp
waitfor --gone $sp --timeout 3m >> "$OUT/run.log" 2>&1 || kill -KILL $sp
wait $sp
kill $smi
grep '^{' "$OUT/ab.jsonl" | cut -c1-200 | sed "s/^/$NAME /" >> "$CD/campaign.log"
echo "EOS-CELL-DONE $NAME" >> "$CD/campaign.log"
