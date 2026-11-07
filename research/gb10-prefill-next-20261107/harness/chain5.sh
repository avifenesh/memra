#!/usr/bin/env bash
# Runs ON unit2 under the lease: the exactness battery on the stacked binary (gdn4) against main,
# with every lever rolled back as the off arm, then the served main-vs-lane campaign.
set -u
D=$HOME/.cache/memra-gb10-next
M=$(ls -d "$D"/bins/main-6f5e03e74a)
L=$(ls -d "$D"/bins/gdn4-* | tail -n 1)
R=$D/runs/gdn5
mkdir -p "$R"
echo "main=$M lane=$L" > "$R/chain.txt"
OFF_ENV="MEMRA_GDN_TRAFFIC=0 MEMRA_PRIME_NORM_FUSE=0 MEMRA_DP4A_TILED=0 MEMRA_FFN_GATEUP_FUSE=0" \
  bash "$D/box/gates_next2.sh" "$L" "$M" "$R/gates" "$D/tree" > "$R/gates.out" 2>&1
echo "gates done" >> "$R/chain.txt"
bash "$D/box/served_ab.sh" served-unit2-final "$M" "$L" "B L L B B L" > "$R/served.out" 2>&1
echo CHAIN5-DONE >> "$R/chain.txt"
