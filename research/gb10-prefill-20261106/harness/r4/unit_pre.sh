#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: lever 5 scouting, the host time before a cold 131k prime's
# first kernel: one profiled served request (osrt + CPU sampling), then pre_window.py on the trace.
set -u
D=$HOME/.cache/memra-gb10-r4
OUT=$D/runs/pre-131k-unit1
BIN=$(ls -d "$D"/bins/gnf-* | tail -n 1)
rm -rf "$OUT"; mkdir -p "$OUT"
bash "$D/box/profile_pre.sh" "$OUT" "$BIN" "" 131072 > "$OUT/profile.out" 2>&1
python3 "$D/box/pre_window.py" "$OUT/prof.sqlite" 0.8 > "$OUT/pre-window.txt" 2>&1
echo "PRE-DONE rc=$?" >> "$OUT/pre-window.txt"
