#!/bin/bash
# 1M serving-config receipt cell, box B (coordinator order: recipe + MEMRA_MLA_TC_PREFILL=1
# + MEMRA_TIMEOUT_MS_MAX per the 1m-demo lane). Placement = THIS window's 3-card resident
# recipe (cards 0/1/2; card 3 is the acceptance-probe co-tenant), NOT the demo's PP4:
# the cell measures whether the SERVING shape is a 1M product. Named deviations:
#  - primeprobe runs from a port-patched copy (18500 -> 18400): the demo port 18500 is
#    claimed by the card3 co-tenant lane on this box.
#  - deep prime is the VENDOR-DEFAULT shape (serving law: the traffic shape is the
#    receipt shape; the demo banked greedy/vendor prefill agreement to 0.01% at 1M).
#  - MEMRA_MAX_SESSIONS=4 (the recipe admission shape, not the demo's capacity pin of 1).
set -u
R=/root/out-1m-b
BIN=/root/memra/target/release/memra-server
C=/root/corpus-1m/corpus-1m.txt
CHARS=4282700   # the demo's chars target: ~1.04M tokens inside the 1,048,576 window
PP=/root/primeprobe-18400.py
mkdir -p "$R"
{
date -u +%FT%TZ
echo "=== BOOT 1M: 3-card resident recipe + MLA_TC + CTX=1048576 ==="
bash /root/serve-scoped-b.sh "$BIN" "$R/serve-1m-b.log" 18400 \
  MEMRA_SPILL_STATS=1 MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_GROUPED_PREFILL=1 \
  MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 \
  MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 CUDA_VISIBLE_DEVICES=0,1,2 \
  MEMRA_COMPAT=openai MEMRA_MODELS=zai/glm-5.3-flash=/root/models/glm53-nvfp4 \
  MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 \
  MEMRA_CTX=1048576 MEMRA_PREFIX_CACHE_MB=0 MEMRA_TIMEOUT_MS_MAX=64800000 \
  MEMRA_MLA_TC_PREFILL=1 || { echo BOOT-FAILED; exit 1; }
grep -n "resident-experts decision" "$R/serve-1m-b.log" | tee "$R/residency-decisions.txt"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | tee "$R/vram-at-ready.txt"

echo; echo "=== W1K warm rung (greedy, 32 tok): slab sanity + session-1 VRAM ==="
python3 "$PP" W1K "$C" 4200 32 greedy "$R/w1k.json"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$R/vram-after-w1k.txt"

echo; echo "=== DEEP PRIME STARTING (coordinator warning ping) ==="
date -u +%FT%TZ
echo "$(date -u +%FT%TZ) lane/glm5-prefix-latent window: 1M DEEP PRIME STARTING (vendor-default, chars=$CHARS, 3-card recipe + MLA_TC, timeout override armed) — FINAL cell of the window" >> /root/BOX-QUEUE.md
bash /root/memra/research/glm53-flash-bringup-20260827/1m-demo-20260829/vramwatch.sh "$R/vram-1m.csv" 20 &
VW=$!
echo "vramwatch pid $VW"
python3 "$PP" R1M "$C" "$CHARS" 256 vendor "$R/r1m-vendor.json"
date -u +%FT%TZ
kill $VW 2>/dev/null && echo "vramwatch stopped"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee "$R/vram-after-1m.txt"

echo "--- per-card VRAM peak over the prime ---"
python3 - << 'PYEOF'
import csv
rows = [r for r in csv.reader(open("/root/out-1m-b/vram-1m.csv")) if r and r[0] != "ts"]
for g in range(4):
    try:
        print(f"  gpu{g}: peak {max(int(r[g+1]) for r in rows)} MiB")
    except Exception as e:
        print(f"  gpu{g}: n/a ({e})")
PYEOF
echo "--- error census of the serve log ---"
grep -cE "out.of.memory|panicked|CUDA_ERROR|engine-error" "$R/serve-1m-b.log" || true
echo "--- serve log tail (meter/admission filtered) ---"
grep -vE "meter|admission" "$R/serve-1m-b.log" | tail -8
date -u +%FT%TZ
echo CELL-DONE
} 2>&1 | tee "$R/cell-transcript.txt"
