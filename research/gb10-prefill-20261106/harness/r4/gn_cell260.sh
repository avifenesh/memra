#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one 260k-only cell of lever 4's served A/B (the anomaly extension) (the GDN out-norm
# inside the fused K4+K5 pass). B = bins/gnb-* (the lever's base), G = bins/gn-* (the lever), both
# built by unit_gn.sh build. usage: gn_cell.sh CAMPAIGN NAME ARM   (ARM = B or G)
set -u
D=$HOME/.cache/memra-gb10-r4
CAMP=$1; NAME=$2; ARM=$3
CD=$D/runs/$CAMP; mkdir -p "$CD"
B=$(ls -d "$D"/bins/gnb-* 2>/dev/null | tail -n 1); G=$(ls -d "$D"/bins/gn-* 2>/dev/null | tail -n 1)
[ -n "$B" ] && [ -n "$G" ] || { echo "GN-CELL-NOBIN" >> "$CD/campaign.log"; exit 1; }
bin=$B; [ "$ARM" = G ] && bin=$G
echo "$(date -Is) cell $NAME bin=$bin" >> "$CD/campaign.log"
bash "$D/box/run_arm.sh" "$CD/$NAME" "$bin" "" "260000" "1" >> "$CD/campaign.log" 2>&1
grep '^{' "$CD/$NAME/run.log" | sed "s/^/$ARM /" >> "$CD/campaign.log"
echo "GN-CELL-DONE $NAME" >> "$CD/campaign.log"
