#!/usr/bin/env bash
# Runs ON unit 1 under the lease: the norm-skip lever. Builds B = WS_SHA (the stack below it: main +
# #1108 + #1110) and N = the ns bundle's HEAD, then served B N N B B N at 32k / 131k / 260k, then
# kernel-check, argmax, run-spec and the prime-path logits B vs N at 16k / 33k.
# usage: unit_ns.sh WS_SHA
set -u
WS=$1
D=$HOME/.cache/memra-gb10-r4
git -C "$D/tree" fetch -q origin main || exit 1
git -C "$D/tree" fetch -q "$D/ns.bundle" "HEAD:refs/heads/r4-lane-ns" || { echo NS-FETCH-FAIL > "$D/logs/unit_ns.txt"; exit 1; }
NS=$(git -C "$D/tree" rev-parse refs/heads/r4-lane-ns)
git -C "$D/tree" checkout -q --detach "$WS" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" wsr > "$D/logs/build-wsr.out" 2>&1
git -C "$D/tree" checkout -q --detach "$NS" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" ns > "$D/logs/build-ns.out" 2>&1
B=$(ls -d "$D"/bins/wsr-"${WS:0:10}" 2>/dev/null); N=$(ls -d "$D"/bins/ns-"${NS:0:10}" 2>/dev/null)
echo "B=$B N=$N" > "$D/logs/unit_ns.txt"
[ -n "$B" ] && [ -n "$N" ] || { echo UNIT-NS-BUILD-FAIL >> "$D/logs/unit_ns.txt"; exit 1; }
# the exactness gates first (a fail-closed refusal or a logits diff stops the campaign)
bash "$D/box/gates_ws.sh" "$N" "$B" "$D/runs/gates-ns-unit1" "$D/tree" > "$D/logs/gates-ns-unit1.out" 2>&1
if grep -q "DIFFER" "$D/runs/gates-ns-unit1/gates.log" || ! grep -q "kernel-check rc=0" "$D/runs/gates-ns-unit1/gates.log"; then
  echo "UNIT-NS-GATES-FAIL" >> "$D/logs/unit_ns.txt"; exit 1
fi
CD=$D/runs/ns-ab-unit1
bash "$D/box/campaign.sh" "$CD" "32768 131072 260000" "1 1 1" "B|$B|" "N|$N|" "N|$N|" "B|$B|" "B|$B|" "N|$N|" > "$D/logs/ns-ab-unit1.out" 2>&1
python3 "$D/box/summarize.py" "$CD" > "$CD/summary.txt" 2>&1
echo UNIT-NS-DONE >> "$D/logs/unit_ns.txt"
