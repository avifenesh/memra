#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: round-4 tree at origin/main + warm target, build main, then
# one cold 131k served prefill under Nsight Systems and the per-part / gap / copy summaries.
# usage: profile_main.sh [REPO] [WARM_TARGET]
set -u
D=$HOME/.cache/memra-gb10-r4
REPO=${1:-$HOME/projects/memra-cells-repo}; WARM=${2:-$HOME/.cache/memra-base-target}
mkdir -p "$D/logs" "$D/bins" "$D/runs"
bash "$D/box/setup_tree.sh" "$REPO" "$WARM" > "$D/logs/setup.log" 2>&1
git -C "$D/tree" fetch -q origin main
git -C "$D/tree" checkout -q --detach origin/main
git -C "$D/tree" log --oneline -1 >> "$D/logs/setup.log"
JOBS=${JOBS:-16} bash "$D/box/build.sh" main > "$D/logs/build-main.out" 2>&1
B=$(ls -d "$D"/bins/main-* 2>/dev/null | tail -n 1)
echo "bin $B" >> "$D/logs/setup.log"
[ -n "$B" ] || { echo PROFILE-MAIN-BUILD-FAIL >> "$D/logs/setup.log"; exit 1; }
OUT=$D/runs/profile-main-131k
bash "$D/box/profile_arm.sh" "$OUT" "$B" "" 131072 > "$D/logs/profile.out" 2>&1
python3 "$D/box/parts.py" "$OUT/prof.sqlite" --json "$OUT/parts.json" > "$OUT/parts.txt" 2>&1
python3 "$D/box/gaps.py" "$OUT/prof.sqlite" > "$OUT/gaps.txt" 2>&1
python3 "$D/box/memcpy.py" "$OUT/prof.sqlite" > "$OUT/memcpy.txt" 2>&1
TTFT=$(python3 -c 'import json,sys; print([json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]["ttft_s"])' "$OUT/pf.jsonl")
python3 "$D/box/parts_last.py" "$OUT/prof.sqlite" "$TTFT" --json "$OUT/parts-last.json" > "$OUT/parts-last.txt" 2>&1
python3 "$D/box/gaps_api.py" "$OUT/prof.sqlite" "$TTFT" > "$OUT/gaps-api.txt" 2>&1
python3 "$D/box/shapes_last.py" "$OUT/prof.sqlite" "$TTFT" fa_prefill_qw_h2p mul_mat_f8f4v3 mul_mat_f8f4v2 f8f4v2_image_kernel > "$OUT/shapes.txt" 2>&1
echo PROFILE-MAIN-DONE >> "$D/logs/setup.log"
