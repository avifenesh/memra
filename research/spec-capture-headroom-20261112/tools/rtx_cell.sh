#!/usr/bin/env bash
# Rig (RTX 5090 Laptop GPU, under the caller's gpulock lease): one served cell of memra#1221's
# no-regression check. A fresh server of BIN serves Qwen3.8-27B (NVFP4+Q5_K trunk + MTP draft, the
# GB10 cells' files) at MEMRA_CTX=32768 with MEMRA_TICK_TRACE=1 (tsrun.py), then specpub_ab.py
# --set 28k (g1k, then a cold 28,000-token greedy request, both to EOS), then the server stops.
# CTX and SET override the context (32768) and the cold request (28k).
# usage: [CTX=N] [SET=28k|60k|100k] rtx_cell.sh CAMPAIGN NN-ARM BIN_PATH   -> ~/.cache/memra-1221/runs/CAMPAIGN/NN-ARM
set -u
D=$HOME/.cache/memra-1221
CAMP=$1; NAME=$2; BIN=$3; ARM=${NAME#*-}
CD=$D/runs/$CAMP; OUT=$CD/$NAME; mkdir -p "$OUT"
M=/data/ai-ml/models/q38-gguf
PORT=8191
{ date -Is; echo "$NAME"; sha256sum "$BIN" | sed "s|$HOME|~|";
  nvidia-smi --query-gpu=name,driver_version,power.limit,temperature.gpu --format=csv,noheader; uptime; } > "$OUT/guard.log"
MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:$PORT MEMRA_CTX=${CTX:-32768} MEMRA_KV_HOST_MB=0 \
MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_MTP_DRAFT="$M/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf" \
MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_TICK_TRACE=1 \
  python3 "$D/tools/tsrun.py" "$BIN" > "$OUT/server.log" 2>&1 &
sp=$!
waitfor --until "curl -sf -o /dev/null http://127.0.0.1:$PORT/readyz && exit 0 || exit 1" --fail-if "! kill -0 $sp" --every 5s --timeout 10m > "$OUT/run.log" 2>&1 \
  || { kill -TERM "$sp"; wait $sp; echo "CELL-FAIL $NAME (not ready)" >> "$CD/campaign.log"; exit 1; }
BENCH_BASE=http://127.0.0.1:$PORT python3 "$D/tools/specpub_ab.py" "$OUT/ab.jsonl" --set ${SET:-28k} --tag "$NAME" >> "$OUT/run.log" 2>&1
kill -TERM "$sp"
waitfor --gone "$sp" --timeout 3m >> "$OUT/run.log" 2>&1 || kill -KILL "$sp"
wait $sp
grep -E '\[(tick-spec|prefix-cache|spec|device-trim|admit-trim|admit-cal)\]' "$OUT/server.log" > "$OUT/trace.txt"
gzip -f "$OUT/server.log"
grep '^{' "$OUT/ab.jsonl" | cut -c1-260 | sed "s/^/$ARM /" >> "$CD/campaign.log"
grep -c '^{' "$OUT/ab.jsonl" | grep -qx 2 || { echo "CELL-FAIL $NAME (rows missing)" >> "$CD/campaign.log"; exit 1; }
echo "CELL-DONE $NAME" >> "$CD/campaign.log"
