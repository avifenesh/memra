#!/bin/bash
# Phase 7: 1M, the config the receipts now justify. What phases 3-6 established, one arm
# per boot: the "slabs" are HOST-pinned staging (MEMRA_ST_PINNED), never device residency;
# the 165-172 tok/s prefill of phases 3-5 came from the expert SLRU arena acting as
# de-facto device residency (it grows on demand and the auto cap is 0.85 of free VRAM,
# which is also what left no room for the 1M request's ~38 GB of upfront allocations on
# the last-stage card and OOM'd every 1M attempt); phase 6's SLOTS=256 starved the
# fused-epi SLRU arm below 3*n_used and fell closed to the sequential loop at ~40 tok/s.
# The demonstrated equilibrium: MEMRA_MOE_SLOTS=12000 caps any card's arena at ~52 GB
# while each stage's working set (10/13/13/6 expert layers under MEMRA_PP_SPLITS=13,26,39)
# grows it only to ~39/51/51/23 GB, leaving the tail-stage card (primary engine + output
# head + the 17.1 GB whole-prime hidden stack + latent planes + the 6.7 GB kpool score
# transient) ~35 GB of headroom at 1M.
set -u
R=$HOME/lane-1mdemo-vast-20260829
BIN=$HOME/wt-1mdemo/target/release/memra-server
C=$R/corpus-1m.txt
CHARS=4282700   # ratio drifts up with corpus depth (4.089 at 258k, 4.113 at 526k); target ~1,040,000 tokens inside the 1,048,576 window
{
  date -u +%FT%TZ
  echo "=== BOOT PP4-1M splits 13,26,39 ==="
  bash $R/serve-1m.sh PP4-1M-G "$BIN" MEMRA_CTX=1048576 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PP_SPLITS=13,26,39 MEMRA_MOE_SLOTS=12000 \
    MEMRA_MOE_RESIDENT_HEADROOM_GB=36 || exit 1
  bash $R/vramwatch.sh $R/phase7-vram.csv 20 &
  VW=$!
  echo "vramwatch pid $VW"

  echo; echo "=== warm rung (1k, greedy): slab populate + sanity on the splits ==="
  python3 $R/primeprobe.py W1K "$C" 4200 32 greedy $R/phase7-W1K.json
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "=== R1M greedy (chars=$CHARS) ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R1M "$C" "$CHARS" 256 greedy $R/phase7-R1M.json
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "--- serve log tail ---"
  grep -vE "meter|admission" $R/serve-PP4-1M-G.log | tail -8

  if python3 -c "import json,sys; j=json.load(open('$R/phase7-R1M.json')); sys.exit(0 if j['status']==200 and not j['error'] else 1)"; then
    echo; echo "=== R1M vendor-default sampled twin ==="
    date -u +%FT%TZ
    python3 $R/primeprobe.py R1M-V "$C" "$CHARS" 256 vendor $R/phase7-R1M-V.json
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  fi

  kill $VW 2>/dev/null && echo "vramwatch $VW stopped"
  echo "--- per-card VRAM peak over phase 5 ---"
  python3 - <<'PYEOF'
import csv
rows = [r for r in csv.reader(open("/root/lane-1mdemo-vast-20260829/phase7-vram.csv")) if r and r[0] != "ts"]
for g in range(4):
    print(f"  gpu{g}: peak {max(int(r[g+1]) for r in rows)} MiB")
PYEOF
  echo "--- error census of the serve log ---"
  grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error" $R/serve-PP4-1M-G.log || true
  date -u +%FT%TZ
  echo PHASE5DONE
} 2>&1 | tee $R/11-phase7.txt
