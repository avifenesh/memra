#!/usr/bin/env bash
# Served cells, one gpulock lease each, interleaved main/fix.
set -u
cd "$HOME/.cache/memra-1217" || exit 1
R=$HOME/.cache/memra-1217/served
mkdir -p "$R"
cell() { # name binary mode env...
  local name=$1 bin=$2 mode=$3; shift 3
  echo "$(date '+%F %T') start $name" >> "$R/cells.txt"
  gpulock run --wait 3h --ttl 20m --for "memra #1217 served cell $name" -- \
    python3 scripts/serve_cell.py "$PWD/bin/$bin" "$R/$name" "$mode" "$@" > "$R/$name.out" 2>&1
  echo "$(date '+%F %T') done $name rc=$? $(tail -c 600 "$R/$name.out" | tr '\n' ' ')" >> "$R/cells.txt"
}
SLRU=MEMRA_MOE_RESIDENT=0
SPILL="MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=4000"
cell fix-res-1 memra-server-fix pair
cell main-res-1 memra-server-main pair
# SLRU, idle card: M F F M M F
cell main-slru-1 memra-server-main pair $SLRU
cell fix-slru-1 memra-server-fix pair $SLRU
cell fix-slru-2 memra-server-fix pair $SLRU
cell main-slru-2 memra-server-main pair $SLRU
cell main-slru-3 memra-server-main pair $SLRU
cell fix-slru-3 memra-server-fix pair $SLRU
# SLRU under spill (4000 slots): F M M F F M
cell fix-spill-1 memra-server-fix pair $SPILL
cell main-spill-1 memra-server-main pair $SPILL
cell main-spill-2 memra-server-main pair $SPILL
cell fix-spill-2 memra-server-fix pair $SPILL
cell fix-spill-3 memra-server-fix pair $SPILL
cell main-spill-3 memra-server-main pair $SPILL
# two sessions under spill: M F F M, then one traced fix run for the lock hold
cell main-itl-1 memra-server-main itl $SPILL
cell fix-itl-1 memra-server-fix itl $SPILL
cell fix-itl-2 memra-server-fix itl $SPILL
cell main-itl-2 memra-server-main itl $SPILL
cell fix-itl-trace memra-server-fix itl $SPILL MEMRA_MOE_SLRU_PAIRS_TRACE=1
echo SERVED-DONE >> "$R/cells.txt"
