#!/usr/bin/env bash
# Runs ON unit1 under the lease: build the fill lane (incremental on main's target), then a served
# B V V B at 32k + 131k (fresh boot per cell). usage: unit_b.sh
set -u
D=$HOME/.cache/memra-gb10-r4
NOTOUCH=1 bash "$D/box/lane_build.sh" fill > "$D/logs/build-fill.out" 2>&1
B=$D/bins/main-d6ef68f5a1
V=$(ls -d "$D"/bins/fill-* 2>/dev/null | tail -n 1)
[ -n "$V" ] || { echo UNIT-B-BUILD-FAIL > "$D/logs/unit_b.txt"; exit 1; }
echo "B=$B V=$V" > "$D/logs/unit_b.txt"
CD=$D/runs/fill-ab-unit1
bash "$D/box/campaign.sh" "$CD" "32768 131072" "1 1" "B|$B|" "V|$V|" "V|$V|" "B|$B|" > "$D/logs/fill-ab-unit1.out" 2>&1
python3 "$D/box/summarize.py" "$CD" > "$CD/summary.txt" 2>&1
echo UNIT-B-DONE >> "$D/logs/unit_b.txt"
