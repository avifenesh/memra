#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: tree at origin/main + warm target, build main, then one cold
# 131k served prefill under Nsight Systems and the per-part / gap / copy summaries.
# usage: profile_main.sh REPO WARM_TARGET
set -u
D=$HOME/.cache/memra-gb10-next
REPO=$1; WARM=$2
mkdir -p "$D/logs" "$D/bins" "$D/runs"
bash "$D/box/setup_tree.sh" "$REPO" "$WARM" > "$D/logs/setup.log" 2>&1
git -C "$D/tree" fetch -q origin main
git -C "$D/tree" checkout -q --detach origin/main
git -C "$D/tree" log --oneline -1 >> "$D/logs/setup.log"
JOBS=8 bash "$D/box/build.sh" main > "$D/logs/build-main.out" 2>&1
B=$(ls -d "$D"/bins/main-* | tail -n 1)
echo "bin $B" >> "$D/logs/setup.log"
OUT=$D/runs/profile-main-131k
bash "$D/box/profile_arm.sh" "$OUT" "$B" "" 131072 > "$D/logs/profile.out" 2>&1
python3 "$D/box/parts.py" "$OUT/prof.sqlite" --json "$OUT/parts.json" > "$OUT/parts.txt" 2>&1
python3 "$D/box/gaps.py" "$OUT/prof.sqlite" > "$OUT/gaps.txt" 2>&1
python3 "$D/box/memcpy.py" "$OUT/prof.sqlite" > "$OUT/memcpy.txt" 2>&1
echo PROFILE-MAIN-DONE >> "$D/logs/setup.log"
