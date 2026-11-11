#!/usr/bin/env bash
# The base binary's staged program on the non-resident placement (slow), run while the fix builds.
set -u
D=$HOME/.cache/memra-1225
cd "$D" || exit 1
R=$D/step3/served
mkdir -p "$R"
cell() { local name=$1 bin=$2 mode=$3; shift 3
  echo "$(date '+%F %T') start $name" >> "$R/cells.txt"
  gpulock run --wait 3h --ttl 60m --for "memra #1225 served cell $name" -- \
    python3 scripts/serve_cell.py "$D/bin/$bin" "$R/$name" "$mode" "$@" > "$R/$name.out" 2>&1
  echo "$(date '+%F %T') done $name rc=$? $(tail -c 700 "$R/$name.out" | tr '\n' ' ')" >> "$R/cells.txt"
}
cell base-slru-1 memra-server-base pair MEMRA_MOE_RESIDENT=0
cell dec-base-slru-1 memra-server-base decode MEMRA_MOE_RESIDENT=0
echo STAGED-DONE >> "$R/cells.txt"
