#!/usr/bin/env bash
# Step 3: the fix on the RTX 5090, one gpulock lease per cell.
set -u
D=$HOME/.cache/memra-1225
cd "$D" || exit 1
# A. placement test, green on the fix: int8-MMA pairs arm and the decode-once dp4a arm
# B. argmax window on gemma-4-26B-A4B board-2048
bash scripts/cells.sh bin/argmax-margin-probe-fix step3/window \
  'res-e:' 'res-b:DECODE_BATCH' \
  'slru-e:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLRU_PAIRS_TRACE=1' 'slru-b:MEMRA_MOE_RESIDENT=0,DECODE_BATCH' \
  'slru2000-e:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLOTS=2000,MEMRA_MOE_SLRU_PAIRS_TRACE=1' \
  'slru400-e:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLOTS=400,MEMRA_MOE_SLRU_PAIRS_TRACE=1' \
  'slru400-b:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLOTS=400,DECODE_BATCH'
# C/D. served cells
R=$D/step3/served
mkdir -p "$R"
cell() { # name binary mode env...
  local name=$1 bin=$2 mode=$3; shift 3
  echo "$(date '+%F %T') start $name" >> "$R/cells.txt"
  gpulock run --wait 3h --ttl 60m --for "memra #1225 served cell $name" -- \
    python3 scripts/serve_cell.py "$D/bin/$bin" "$R/$name" "$mode" "$@" > "$R/$name.out" 2>&1
  echo "$(date '+%F %T') done $name rc=$? $(tail -c 700 "$R/$name.out" | tr '\n' ' ')" >> "$R/cells.txt"
}
SLRU=MEMRA_MOE_RESIDENT=0
# served greedy pair (board-2048, 128 tokens): resident M F F M M F
cell base-res-1 memra-server-base pair
cell fix-res-1 memra-server-fix pair
cell fix-res-2 memra-server-fix pair
cell base-res-2 memra-server-base pair
cell base-res-3 memra-server-base pair
cell fix-res-3 memra-server-fix pair
# SLRU placement: base is the staged program (N=1, slow), fix default cache and spill N=3
cell fix-slru-1 memra-server-fix pair $SLRU
cell fix-spill-1 memra-server-fix pair $SLRU MEMRA_MOE_SLOTS=2000
cell fix-slru-2 memra-server-fix pair $SLRU
cell fix-spill-2 memra-server-fix pair $SLRU MEMRA_MOE_SLOTS=2000
cell fix-slru-3 memra-server-fix pair $SLRU
cell fix-spill-3 memra-server-fix pair $SLRU MEMRA_MOE_SLOTS=2000
# decode rate (short prompt, 256 tokens)
cell dec-base-res-1 memra-server-base decode
cell dec-fix-res-1 memra-server-fix decode
cell dec-fix-slru-1 memra-server-fix decode $SLRU
cell dec-fix-half-1 memra-server-fix decode $SLRU MEMRA_MOE_SLOTS=6000
cell dec-fix-spill-1 memra-server-fix decode $SLRU MEMRA_MOE_SLOTS=2000
cell dec-fix-spill-2 memra-server-fix decode $SLRU MEMRA_MOE_SLOTS=2000
cell dec-fix-half-2 memra-server-fix decode $SLRU MEMRA_MOE_SLOTS=6000
cell dec-fix-slru-2 memra-server-fix decode $SLRU
cell dec-fix-res-2 memra-server-fix decode
cell dec-base-res-2 memra-server-base decode
cell dec-base-res-3 memra-server-base decode
cell dec-fix-res-3 memra-server-fix decode
cell dec-fix-slru-3 memra-server-fix decode $SLRU
cell dec-fix-half-3 memra-server-fix decode $SLRU MEMRA_MOE_SLOTS=6000
cell dec-fix-spill-3 memra-server-fix decode $SLRU MEMRA_MOE_SLOTS=2000
echo STEP3-DONE >> "$R/cells.txt"
