#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1266): one boot of the normal-concurrency
# cell #1173 and #1255 used (normal_cell.py: closed-loop clients at c = 1, 2, 4, 8, 6 requests per
# client, greedy chat, 160 new tokens, streamed; ctx 8,192, 8 sessions) on Qwen3.8-27B NVFP4 + Q5_K
# on the MTP route this change acts on (the masked 32,768-row MTP head, no DFlash drafter).
# usage: normal_gb10.sh CAMPAIGN NAME ARM   -> runs/CAMPAIGN/NAME/
set -u
D=<lane>
CAMP=$1; NAME=$2; BIN=$D/bins/$3/memra-server
OUT=$D/runs/$CAMP/$NAME; mkdir -p "$OUT"
G8=<models>/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
{ date -Is; echo "$NAME"; sha256sum "$BIN" | sed "s|$HOME|~|"; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader; uptime; } > "$OUT/guard.log"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -lms 1000 > "$OUT/gpu.csv" 2>&1 &
smi=$!
export CELL_MODEL=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf CELL_PORT=18266
export PLAN=1,2,4,8 REQS=6 MAXTOK=160 STREAM=1 TAG=$NAME
export EXTRA_ENV="MEMRA_MTP_DRAFT=$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
python3 "$D/box/normal_cell.py" "$OUT" "$BIN" "$NAME" > "$OUT/cell.log" 2>&1; rc=$?
kill $smi 2>/dev/null
{ for pat in 'out of memory' 'NOT parked' 'pooled MTP verify capped' 'pre-cohort reclaim' 'spec-pool] pooled MTP cohort engaged'; do
    echo "$pat=$(grep -a -c -F "$pat" "$OUT/server-$NAME.log")"; done; } > "$OUT/summary.txt"
gzip -f "$OUT/server-$NAME.log"
echo "$(date -Is) NORMAL-DONE $NAME rc=$rc" >> "$D/runs/$CAMP/campaign.log"
exit $rc
