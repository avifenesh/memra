#!/usr/bin/env bash
# Corpus generation campaign, box shape: one RTX PRO 6000, card 0.
# Holds the box GPU lock for the whole run (server + driver are one campaign).
set -euo pipefail
cd "$HOME/models/ornith15"
mkdir -p mtp-train

exec 9>/tmp/memra-gpu.lock
flock -n 9 || { echo "FATAL: /tmp/memra-gpu.lock busy"; exit 1; }

MODEL="$HOME/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-MTP.gguf"
[ -f "$MODEL" ] || { echo "FATAL: model missing: $MODEL"; exit 1; }

export CUDA_VISIBLE_DEVICES=0
export MEMRA_MODELS="ornith15=$MODEL"
export MEMRA_ADDR=127.0.0.1:8094
export MEMRA_SERVE_SPEC=0

"$HOME/memra-src/target/release/memra-server" > mtp-train/server.log 2>&1 &
SRV=$!
echo "$SRV" > mtp-train/server.pid

up=0
for _ in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8094/v1/models >/dev/null 2>&1; then up=1; break; fi
  kill -0 "$SRV" 2>/dev/null || { echo "FATAL: server died"; tail -30 mtp-train/server.log; exit 1; }
  sleep 2
done
[ "$up" = 1 ] || { echo "FATAL: server not healthy after 240s"; kill "$SRV"; exit 1; }
echo "server up pid=$SRV"

RC=0
python3 mtp-train/gen_corpus.py \
  --pack-dir mtp-train/prompts \
  --agentic-dir agentic-prompts \
  --out mtp-train/corpus.jsonl \
  --concurrency 8 > mtp-train/gen-corpus.log 2>&1 || RC=$?

kill "$SRV" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
echo "gen-corpus done rc=$RC"
exit "$RC"
