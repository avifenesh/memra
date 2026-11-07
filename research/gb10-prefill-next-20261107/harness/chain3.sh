#!/usr/bin/env bash
# Runs ON unit2 under the lease: build the lane head, kernel-check synthetic arms, the exactness
# battery against main, then the served main-vs-lane campaign (32k/131k/260k per fresh boot).
set -u
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
R=$D/runs/gdn3
mkdir -p "$R"
# base: main at the lane's merge base
git -C "$D/tree" fetch -q origin main
git -C "$D/tree" checkout -q --detach origin/main
git -C "$D/tree" log --oneline -1 > "$R/base.txt"
JOBS=8 bash "$D/box/build.sh" main > "$R/build-main.out" 2>&1
M=$(ls -d "$D"/bins/main-$(git -C "$D/tree" rev-parse --short=10 HEAD) 2>/dev/null)
echo "main: $(tail -n 1 "$R/build-main.out") bin=$M" > "$R/chain.txt"
[ -n "$M" ] || { echo CHAIN3-MAIN-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
bash "$D/box/lane_build.sh" gdn3 > "$R/build.out" 2>&1
L=$(ls -d "$D"/bins/gdn3-* 2>/dev/null | tail -n 1)
echo "build: $(tail -n 1 "$R/build.out") bin=$L" >> "$R/chain.txt"
[ -n "$L" ] || { echo CHAIN3-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GDN-DB-GQA|ADD-RMS-NORM-E4M3V2|add_rms_norm fused|NVFP4-DP4A-RP-T4" "$R/kc-fast.log" >> "$R/chain.txt"
bash "$D/box/gates_next.sh" "$L" "$M" "$R/gates" "$D/tree" > "$R/gates.out" 2>&1
echo "gates done" >> "$R/chain.txt"
bash "$D/box/served_ab.sh" served-unit2 "$M" "$L" "B L L B B L" > "$R/served.out" 2>&1
echo CHAIN3-DONE >> "$R/chain.txt"
