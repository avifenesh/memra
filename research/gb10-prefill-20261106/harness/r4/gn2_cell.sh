#!/usr/bin/env bash
# Runs ON unit 2 under the caller's lease: one cell of lever 4's unit-2 served A/B (the GDN out-norm
# inside the fused K4+K5 pass). The first call builds B = #1117's head and G = the lever's head from
# gn3.bundle. usage: gn2_cell.sh NAME ARM B_SHA G_SHA   (ARM = B or G)
set -u
D=$HOME/.cache/memra-gb10-r4
NAME=$1; ARM=$2; BS=$3; GS=$4
CD=$D/runs/gn-ab-unit2; mkdir -p "$CD"
if ! ls -d "$D"/bins/gnf2-* > /dev/null 2>&1; then
  git -C "$D/tree" fetch -q origin main
  git -C "$D/tree" fetch -q "$D/gn3.bundle" "HEAD:refs/heads/r4-lane-gn3" || { echo GN-CELL-BUILD-FAIL >> "$CD/campaign.log"; exit 1; }
  git -C "$D/tree" checkout -q --detach "$BS" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" gnb2 > "$D/logs/build-gnb2-unit2.out" 2>&1
  git -C "$D/tree" checkout -q --detach "$GS" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" gnf2 > "$D/logs/build-gnf2-unit2.out" 2>&1
fi
B=$(ls -d "$D"/bins/gnb2-* 2>/dev/null | tail -n 1); G=$(ls -d "$D"/bins/gnf2-* 2>/dev/null | tail -n 1)
[ -n "$B" ] && [ -n "$G" ] || { echo "GN-CELL-BUILD-FAIL" >> "$CD/campaign.log"; exit 1; }
bin=$B; [ "$ARM" = G ] && bin=$G
echo "$(date -Is) cell $NAME bin=$bin" >> "$CD/campaign.log"
bash "$D/box/run_arm.sh" "$CD/$NAME" "$bin" "" "32768 131072 260000" "1 1 1" >> "$CD/campaign.log" 2>&1
grep '^{' "$CD/$NAME/run.log" | sed "s/^/$ARM /" >> "$CD/campaign.log"
echo "GN-CELL-DONE $NAME" >> "$CD/campaign.log"
