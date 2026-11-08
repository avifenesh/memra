#!/usr/bin/env bash
# Runs ON unit 2 under the caller's lease: one cell of the norm-skip campaign; the first call builds
# B = WS_SHA and N = the ns bundle's HEAD. usage: ns_cell.sh NAME ARM WS_SHA   (ARM = B or N)
set -u
D=$HOME/.cache/memra-gb10-r4
NAME=$1; ARM=$2; WS=$3
CD=$D/runs/ns-ab-unit2; mkdir -p "$CD"
if ! ls -d "$D"/bins/ns-* > /dev/null 2>&1; then
  git -C "$D/tree" fetch -q origin main
  git -C "$D/tree" fetch -q "$D/ns.bundle" "HEAD:refs/heads/r4-lane-ns" || { echo NS-CELL-BUILD-FAIL >> "$CD/campaign.log"; exit 1; }
  NS=$(git -C "$D/tree" rev-parse refs/heads/r4-lane-ns)
  git -C "$D/tree" checkout -q --detach "$WS" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" wsr > "$D/logs/build-wsr-unit2.out" 2>&1
  git -C "$D/tree" checkout -q --detach "$NS" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" ns > "$D/logs/build-ns-unit2.out" 2>&1
fi
B=$(ls -d "$D"/bins/wsr-* 2>/dev/null | tail -n 1); N=$(ls -d "$D"/bins/ns-* 2>/dev/null | tail -n 1)
[ -n "$B" ] && [ -n "$N" ] || { echo "NS-CELL-BUILD-FAIL" >> "$CD/campaign.log"; exit 1; }
bin=$B; [ "$ARM" = N ] && bin=$N
echo "$(date -Is) cell $NAME bin=$bin" >> "$CD/campaign.log"
bash "$D/box/run_arm.sh" "$CD/$NAME" "$bin" "" "32768 131072 260000" "1 1 1" >> "$CD/campaign.log" 2>&1
grep '^{' "$CD/$NAME/run.log" | sed "s/^/$ARM /" >> "$CD/campaign.log"
echo "NS-CELL-DONE $NAME" >> "$CD/campaign.log"
