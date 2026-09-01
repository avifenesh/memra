#!/bin/bash
# The wall bracket, ONE boot: merged-head binary (ring ON default), MEMRA_CTX=262144 so every
# rung fits the configured window and a failure is capacity, not admission. Doubling ladder to
# just under the window; the largest rung leaves margin for the chat template so the prompt is
# admitted. This is the honest upper bound for the model card's context statement; the old ~50k
# OOM was the quadratic score plane, and the merged chunked prime is expected to have removed it.
set -u
R=$HOME/lane-ringsizing-vast-20260829
MERGED=${1:?merged-head binary}
{
  echo "================================================================"
  date -u +%FT%TZ
  echo "=== WALL BRACKET: merged head, ring ON (default), MEMRA_CTX=262144 ==="
  bash $R/serve.sh WALL-262K "$MERGED" MEMRA_CTX=262144 || { echo "WALL: LOAD FAILED"; exit 1; }
  python3 $R/wallprobe.py WALL-262K $R/wall-262k.json 8000 16000 32000 64000 128000 250000
  echo "--- vram after ladder ---"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  date -u +%FT%TZ
  echo "WALLDONE"
} 2>&1 | tee $R/03-wall-bracket.txt
