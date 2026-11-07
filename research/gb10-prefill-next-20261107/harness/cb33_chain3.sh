#!/usr/bin/env bash
# Runs ON unit1 (unit 1) under its lease: build main 6f5e03e74a and the PR head (gdn6), the
# exactness battery (every lever rolled back as the off arm, the batched-prime gate), then the
# served confirmation campaign B L L B.
set -u
D=$HOME/.cache/memra-gb10-next
R=$D/runs/unit1-gdn6
mkdir -p "$R"
git -C "$D/tree" fetch -q origin main
git -C "$D/tree" checkout -q --detach 6f5e03e74a
JOBS=8 bash "$D/box/build.sh" main > "$R/build-main.out" 2>&1
M=$(ls -d "$D"/bins/main-6f5e03e74a 2>/dev/null)
bash "$D/box/lane_build2.sh" gdn6 "$D/lane6.bundle" > "$R/build.out" 2>&1
L=$(ls -d "$D"/bins/gdn6-* 2>/dev/null | tail -n 1)
echo "main=$M lane=$L $(tail -n 1 "$R/build-main.out") / $(tail -n 1 "$R/build.out")" > "$R/chain.txt"
[ -n "$M" ] && [ -n "$L" ] || { echo CB33-3-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
OFF_ENV="MEMRA_GDN_TRAFFIC=0 MEMRA_GDN_K45=0 MEMRA_PRIME_NORM_FUSE=0 MEMRA_DP4A_TILED=0 MEMRA_FFN_GATEUP_FUSE=0" \
  bash "$D/box/gates_next2.sh" "$L" "$M" "$R/gates" "$D/tree" > "$R/gates.out" 2>&1
echo "gates done" >> "$R/chain.txt"
bash "$D/box/served_ab.sh" served-unit1-final "$M" "$L" "B L L B" > "$R/served.out" 2>&1
echo CB33-CHAIN3-DONE >> "$R/chain.txt"
