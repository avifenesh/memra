#!/bin/bash
# Phase 2 of the 1M demonstration: ONE boot at MEMRA_CTX=1048576 on the PP4 placement,
# then the ladder on real corpus prefixes of the sha-banked corpus-1m.txt:
#   R1K    sanity at this boot (greedy 64)
#   R131K  timing + tokenizer calibration rung (greedy 128)
#   R1M    the demonstration prime, greedy 256 (decode-at-depth rides the same request)
#   R1M-V  the vendor-default sampled twin (second full prime; second timing receipt)
# vramwatch samples per-card VRAM every 20 s for the whole phase, so the peak is banked.
# The R1M char count is computed from R131K's server-reported prompt_tokens (usage is the
# only honest token count), targeting 1,043,000 prompt tokens inside the 1,048,576 window
# with max_tokens 256 + template margin. An admission refusal retries once at -1.5%.
set -u
R=$HOME/lane-1mdemo-vast-20260829
BIN=$HOME/wt-1mdemo/target/release/memra-server
C=$R/corpus-1m.txt
{
  date -u +%FT%TZ
  echo "=== BOOT: PP4-1M (MEMRA_CTX=1048576, 4 stages, 4 cards) ==="
  bash $R/serve-1m.sh PP4-1M "$BIN" MEMRA_CTX=1048576 CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_PP_STAGES=4 MEMRA_PP_DEVICES=0,1,2,3 MEMRA_MOE_RESIDENT_HEADROOM_GB=36 || exit 1

  bash $R/vramwatch.sh $R/phase2-vram.csv 20 &
  VW=$!
  echo "vramwatch pid $VW"

  echo; echo "=== R1K: sanity rung ==="
  python3 $R/primeprobe.py R1K "$C" 4200 64 greedy $R/phase2-R1K.json

  echo; echo "=== R131K: timing + calibration rung ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R131K "$C" 527000 128 greedy $R/phase2-R131K.json
  date -u +%FT%TZ
  echo "--- vram after R131K ---"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  CHARS=$(python3 - <<'EOF'
import json
j = json.load(open("/root/lane-1mdemo-vast-20260829/phase2-R131K.json"))
pt = j["usage"]["prompt_tokens"]
ratio = j["chars"] / pt
print(int(1043000 * ratio))
EOF
)
  echo; echo "=== R1M: the demonstration prime, greedy (chars=$CHARS from R131K ratio) ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R1M "$C" "$CHARS" 256 greedy $R/phase2-R1M.json
  date -u +%FT%TZ
  if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$R/phase2-R1M.json'))['status']==200 else 1)"; then
    CHARS=$((CHARS * 985 / 1000))
    echo "R1M non-200; ONE retry at chars=$CHARS"
    date -u +%FT%TZ
    python3 $R/primeprobe.py R1M-retry "$C" "$CHARS" 256 greedy $R/phase2-R1M.json
    date -u +%FT%TZ
  fi
  echo "--- vram after R1M ---"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  echo "--- serve log tail after R1M ---"
  tail -25 $R/serve-PP4-1M.log

  echo; echo "=== R1M-V: vendor-default sampled twin ==="
  date -u +%FT%TZ
  python3 $R/primeprobe.py R1M-V "$C" "$CHARS" 256 vendor $R/phase2-R1M-V.json
  date -u +%FT%TZ
  echo "--- vram after R1M-V ---"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

  kill $VW 2>/dev/null && echo "vramwatch $VW stopped"
  echo "--- per-card VRAM peak over phase 2 ---"
  python3 - <<'EOF'
import csv
rows = [r for r in csv.reader(open("/root/lane-1mdemo-vast-20260829/phase2-vram.csv")) if r and r[0] != "ts"]
for g in range(4):
    peak = max(int(r[g+1]) for r in rows)
    print(f"  gpu{g}: peak {peak} MiB")
EOF
  echo "--- OOM/panic census of the serve log ---"
  grep -ciE "out.of.memory|panicked|CUDA_ERROR" $R/serve-PP4-1M.log || true
  date -u +%FT%TZ
  echo PHASE2DONE
} 2>&1 | tee $R/04-phase2.txt
