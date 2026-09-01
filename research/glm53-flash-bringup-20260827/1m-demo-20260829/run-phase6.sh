#!/bin/bash
# Phase 6: 1M with uneven splits AND the SLRU double-hold closed. Phase 5's receipt showed
# every card jump to ~79.5 GB after a 1,108-token warm rung REGARDLESS of the split: the
# expert SLRU auto-sizes its slot arena to 0.85 of free VRAM per device at first dispatch,
# ON TOP of the fully-resident slabs it will never miss into. That arena, not the slabs,
# is what left the last-stage card ~17 GB short of the 1M request's upfront allocations.
# MEMRA_MOE_SLOTS=256 pins the arena to its floor; every expert layer is slab-resident
# (phase-3 slab-populate census: 42 layers x 3 matrices), so the SLRU is a failsafe only.
# Phase 5 (splits alone) is banked as the arm that proves the arena, not the slab split,
# was the wall. Phases 3/4 proved the OOM follows the LAST-STAGE card,
# which is always also the worker's primary engine (the tail, the f32 output head, and the
# whole-prime hidden-stack aggregation all live there), and its ~40 GB slab share left no
# room for the ~35-40 GB the 1M request allocates up front on that card. MEMRA_PP_SPLITS=
# 13,26,39 keeps stages 0-2 at 13 layers and cuts the tail stage to layers 39-45, whose
# expert-slab share is ~23 GB instead of ~40 GB. Placement-only change; the ppN gate holds
# bit-identity at every split.
set -u
R=$HOME/lane-1mdemo-vast-20260829
BIN=$HOME/wt-1mdemo/target/release/memra-server
C=$R/corpus-1m.txt
CHARS=4282700   # ratio drifts up with corpus depth (4.089 at 258k, 4.113 at 526k); target ~1,040,000 tokens inside the 1,048,576 window
{
  date -u +%FT%TZ
  echo "=== BOOT PP4-1M splits 13,26,39 ==="
  bash $R/serve-1m.sh PP4-1M-F "$BIN" MEMRA_CTX=1048576 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_PP_SPLITS=13,26,39 MEMRA_MOE_SLOTS=256 \
    MEMRA_MOE_RESIDENT_HEADROOM_GB=36 || exit 1
  bash $R/vramwatch.sh $R/phase6-vram.csv 20 &
  VW=$!
  echo "vramwatch pid $VW"

  echo; echo "=== warm rung (1k, greedy): slab populate + sanity on the splits ==="
  python3 $R/primeprobe.py W1K "$C" 4200 32 greedy $R/phase6-W1K.json
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  echo; echo "=== R1M greedy (chars=$CHARS) ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R1M "$C" "$CHARS" 256 greedy $R/phase6-R1M.json
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "--- serve log tail ---"
  grep -vE "meter|admission" $R/serve-PP4-1M-F.log | tail -8

  if python3 -c "import json,sys; j=json.load(open('$R/phase6-R1M.json')); sys.exit(0 if j['status']==200 and not j['error'] else 1)"; then
    echo; echo "=== R1M vendor-default sampled twin ==="
    date -u +%FT%TZ
    python3 $R/primeprobe.py R1M-V "$C" "$CHARS" 256 vendor $R/phase6-R1M-V.json
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  fi

  kill $VW 2>/dev/null && echo "vramwatch $VW stopped"
  echo "--- per-card VRAM peak over phase 5 ---"
  python3 - <<'PYEOF'
import csv
rows = [r for r in csv.reader(open("/root/lane-1mdemo-vast-20260829/phase6-vram.csv")) if r and r[0] != "ts"]
for g in range(4):
    print(f"  gpu{g}: peak {max(int(r[g+1]) for r in rows)} MiB")
PYEOF
  echo "--- error census of the serve log ---"
  grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error" $R/serve-PP4-1M-F.log || true
  date -u +%FT%TZ
  echo PHASE5DONE
} 2>&1 | tee $R/10-phase6.txt
