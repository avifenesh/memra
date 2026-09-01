#!/bin/bash
# Phase 4: the 1M rung itself, after phase 3's R1M OOM. Diagnosis from the phase-3 receipts:
# every card sits at a ~82 GB steady state during deep primes, and device 3 carried FOUR
# roles at once — the worker's primary engine, PP stage 3 (the tail: collapse + output norm
# + the f32 output head), AND the whole-prime hidden-stack aggregation buffer (17.1 GB at
# 1.03M tokens, allocated on the primary engine) — peaking 97,241 MiB on a 97,887 MiB card.
# The remap MEMRA_PP_DEVICES=3,1,2,0 is pure placement (PP is bit-identical across devices,
# glm5_hyper_ppn_gate): stage 0 shares the primary card (embed + aggregation), the tail and
# head move to device 0, so no single card carries both the aggregation and the tail.
# Fallback rung: if 1M still OOMs, bank 524k and stop; residency shedding is the next lever.
set -u
R=$HOME/lane-1mdemo-vast-20260829
BIN=$HOME/wt-1mdemo/target/release/memra-server
C=$R/corpus-1m.txt
CHARS=4323600   # ratio 4.1374 chars/token measured at the 258k rung -> ~1,045,000 tokens
{
  date -u +%FT%TZ
  echo "=== BOOT PP4-1M remapped (MEMRA_PP_DEVICES=3,1,2,0) ==="
  bash $R/serve-1m.sh PP4-1M-R "$BIN" MEMRA_CTX=1048576 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=3,1,2,0 MEMRA_MOE_RESIDENT_HEADROOM_GB=36 || exit 1
  bash $R/vramwatch.sh $R/phase4-vram.csv 20 &
  VW=$!
  echo "vramwatch pid $VW"

  echo; echo "=== warm rung (1k, greedy): slab populate + sanity on the remap ==="
  python3 $R/primeprobe.py W1K "$C" 4200 32 greedy $R/phase4-W1K.json

  echo; echo "=== R1M greedy (chars=$CHARS) ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R1M "$C" "$CHARS" 256 greedy $R/phase4-R1M.json
  date -u +%FT%TZ
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "--- serve log tail ---"
  grep -vE "meter|admission" $R/serve-PP4-1M-R.log | tail -8

  if python3 -c "import json,sys; j=json.load(open('$R/phase4-R1M.json')); sys.exit(0 if j['status']==200 and not j['error'] else 1)"; then
    echo; echo "=== R1M vendor-default sampled twin ==="
    date -u +%FT%TZ
    python3 $R/primeprobe.py R1M-V "$C" "$CHARS" 256 vendor $R/phase4-R1M-V.json
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  else
    echo; echo "=== R1M failed on the remap too; bank the 524k bracket rung instead ==="
    date -u +%FT%TZ
    python3 $R/primeprobe.py R524K "$C" 2161700 128 greedy $R/phase4-R524K.json
    date -u +%FT%TZ
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  fi

  kill $VW 2>/dev/null && echo "vramwatch $VW stopped"
  echo "--- per-card VRAM peak over phase 4 ---"
  python3 - <<'EOF'
import csv
rows = [r for r in csv.reader(open("/root/lane-1mdemo-vast-20260829/phase4-vram.csv")) if r and r[0] != "ts"]
for g in range(4):
    print(f"  gpu{g}: peak {max(int(r[g+1]) for r in rows)} MiB")
EOF
  echo "--- error census of the serve log ---"
  grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error" $R/serve-PP4-1M-R.log || true
  date -u +%FT%TZ
  echo PHASE4DONE
} 2>&1 | tee $R/08-phase4.txt
