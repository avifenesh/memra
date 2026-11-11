#!/usr/bin/env bash
# Step 1: does gemma-4-26B-A4B's output depend on the expert placement? One lease per cell.
set -u
D=$HOME/.cache/memra-1225
cd "$D" || exit 1
bash scripts/cells.sh bin/argmax-margin-probe-base step1/window \
  'res-e:' 'slru-e:MEMRA_MOE_RESIDENT=0' 'res-b:DECODE_BATCH' 'slru-b:MEMRA_MOE_RESIDENT=0,DECODE_BATCH'
R=$D/step1/served
mkdir -p "$R"
cell() { # name binary env...
  local name=$1 bin=$2; shift 2
  echo "$(date '+%F %T') start $name" >> "$R/cells.txt"
  gpulock run --wait 3h --ttl 60m --for "memra #1225 served cell $name" -- \
    python3 scripts/serve_cell.py "$D/bin/$bin" "$R/$name" pair "$@" > "$R/$name.out" 2>&1
  echo "$(date '+%F %T') done $name rc=$? $(tail -c 600 "$R/$name.out" | tr '\n' ' ')" >> "$R/cells.txt"
}
cell base-res memra-server-base
cell base-slru memra-server-base MEMRA_MOE_RESIDENT=0
echo STEP1-DONE >> "$R/cells.txt"
