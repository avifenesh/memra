#!/usr/bin/env bash
# Runs ON unit2 under the lease: the exactness battery on the glue-2 binary against the #1089 head
# binary (gdn6), with every lever rolled back as the off arm.
set -u
D=$HOME/.cache/memra-gb10-next
M=$(ls -d "$D"/bins/gdn6-* | tail -n 1)
L=$(ls -d "$D"/bins/glue2-* | tail -n 1)
R=$D/runs/glue2-gates
mkdir -p "$R"
echo "base=$M lane=$L" > "$R/chain.txt"
OFF_ENV="MEMRA_GDN_TRAFFIC=0 MEMRA_GDN_K45=0 MEMRA_PRIME_NORM_FUSE=0 MEMRA_DP4A_TILED=0 MEMRA_FFN_GATEUP_FUSE=0" \
  bash "$D/box/gates_next2.sh" "$L" "$M" "$R/gates" "$D/tree" > "$R/gates.out" 2>&1
echo CHAIN8-DONE >> "$R/chain.txt"
