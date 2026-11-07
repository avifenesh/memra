#!/usr/bin/env bash
# Runs ON a GB10 host under the lease, after #1092 is on main: build origin/main (main2) and the PR
# head (final) from origin, kernel-check the final head's synthetic arms, then the served campaign
# B L L B (B = BASE, L = the PR head); the sm_121a cell record against BASE runs first.
# usage: chain15.sh PR_SHA BASE_SHA CAMPAIGN_NAME [ORDER]
set -u
SHA=$1; BASE=$2; CNAME=$3; ORDER=${4:-"B L L B"}
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
R=$D/runs/$CNAME-chain
mkdir -p "$R"
git -C "$D/tree" fetch -q origin main perf/gb10-prime-gdn-glue fix/grid-law-tests-and-self-yield || { echo CHAIN15-FETCH-FAIL > "$R/chain.txt"; exit 1; }
MS=$(git -C "$D/tree" rev-parse --short=10 "$BASE")
echo "main=$MS pr=$SHA" > "$R/chain.txt"
bash "$D/box/rm_cells_out.sh" > /dev/null 2>&1
bash "$D/box/cells_gb10d.sh" "$SHA" perf/gb10-prime-gdn-glue unit2 "$BASE" > "$D/logs/cells-gb10-3.out" 2>&1
echo "cells rc=$? $(tail -n 1 "$D/logs/cells-gb10-3.out")" >> "$R/chain.txt"
git -C "$D/tree" checkout -q --detach "$BASE" && JOBS=12 bash "$D/box/build.sh" main2 > "$R/build-main.out" 2>&1
git -C "$D/tree" checkout -q --detach "$SHA" && JOBS=12 bash "$D/box/build.sh" final > "$R/build-final.out" 2>&1
M=$(ls -d "$D"/bins/main2-"$MS" 2>/dev/null)
L=$(ls -d "$D"/bins/final-"${SHA:0:10}" 2>/dev/null)
echo "bins main=$M final=$L $(tail -n 1 "$R/build-main.out") / $(tail -n 1 "$R/build-final.out")" >> "$R/chain.txt"
[ -n "$M" ] && [ -n "$L" ] || { echo CHAIN15-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GDN-CONV-L2|Q-SPLIT-NORM-ROPE|FAIL" "$R/kc-fast.log" >> "$R/chain.txt"
bash "$D/box/served_ab.sh" "$CNAME" "$M" "$L" "$ORDER" > "$D/logs/$CNAME.out" 2>&1
echo "served done" >> "$R/chain.txt"
echo CHAIN15-DONE >> "$R/chain.txt"
