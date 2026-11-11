#!/usr/bin/env bash
# Final confirmation cells on the rebased head (probe/server -fix) against main (-main), one lease
# per cell. Reduced set (the rig is shared): window SLRU eager/batch and the chunked floor, and one
# served pair and decode request per placement for the text. Timing N=3 is step3's.
set -u
D=$HOME/.cache/memra-1225
cd "$D" || exit 1
bash scripts/cells.sh bin/argmax-margin-probe-fix final/window \
  'slru-e:MEMRA_MOE_RESIDENT=0' 'slru-b:MEMRA_MOE_RESIDENT=0,DECODE_BATCH' \
  'slru360-e:MEMRA_MOE_RESIDENT=0,MEMRA_MOE_SLOTS=360,MEMRA_MOE_SLRU_PAIRS_TRACE=1'
R=$D/final/served
mkdir -p "$R"
cell() { local name=$1 bin=$2 mode=$3; shift 3
  echo "$(date '+%F %T') start $name" >> "$R/cells.txt"
  gpulock run --wait 3h --ttl 60m --for "memra #1225 final served cell $name" -- \
    python3 scripts/serve_cell.py "$D/bin/$bin" "$R/$name" "$mode" "$@" > "$R/$name.out" 2>&1
  echo "$(date '+%F %T') done $name rc=$? $(tail -c 700 "$R/$name.out" | tr '\n' ' ')" >> "$R/cells.txt"
}
SLRU=MEMRA_MOE_RESIDENT=0
cell main-res-1 memra-server-main pair
cell fix-res-1 memra-server-fix pair
cell fix-slru-1 memra-server-fix pair $SLRU
cell fix-spill-1 memra-server-fix pair $SLRU MEMRA_MOE_SLOTS=2000
cell dec-main-res-1 memra-server-main decode
cell dec-fix-res-1 memra-server-fix decode
cell dec-fix-slru-1 memra-server-fix decode $SLRU
cell dec-fix-spill-1 memra-server-fix decode $SLRU MEMRA_MOE_SLOTS=2000
echo FINAL-DONE >> "$R/cells.txt"
