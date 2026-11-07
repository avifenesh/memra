#!/usr/bin/env bash
# Runs ON unit2 under the lease, after chain11: one served pair (B = the #1089 binary gdn6, L = the
# second-pass binary k232), all three lengths per fresh boot.
set -u
D=$HOME/.cache/memra-gb10-next
waitfor --until "grep -q CHAIN11-DONE $D/runs/glue3-gates/chain.txt" --every 20s --timeout 40m > /dev/null 2>&1 || { echo CHAIN12-NO-CHAIN11; exit 1; }
M=$(ls -d "$D"/bins/gdn6-* | tail -n 1)
L=$(ls -d "$D"/bins/k232-* | tail -n 1)
bash "$D/box/served_ab.sh" served-glue3-unit2-pair "$M" "$L" "B L" > "$D/logs/served-glue3-pair.out" 2>&1
echo CHAIN12-DONE >> "$D/runs/glue3-gates/chain.txt"
