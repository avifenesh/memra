#!/usr/bin/env bash
# Runs ON unit2 (unit 2) under the lease: build the PR head (gdn6), kernel-check synthetic arms,
# then the served main-vs-lane campaign (32k/131k/260k per fresh boot, B L L B B L).
set -u
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
M=$(ls -d "$D"/bins/main-6f5e03e74a)
R=$D/runs/gdn6
mkdir -p "$R"
bash "$D/box/lane_build2.sh" gdn6 "$D/lane6.bundle" > "$R/build.out" 2>&1
L=$(ls -d "$D"/bins/gdn6-* 2>/dev/null | tail -n 1)
echo "main=$M lane=$L $(tail -n 1 "$R/build.out")" > "$R/chain.txt"
[ -n "$L" ] || { echo CHAIN6-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GDN-DB-GQA|GATED-RMSNORM-E4M3V2|ADD-RMS-NORM-E4M3V2|NVFP4-DP4A-RP-T4" "$R/kc-fast.log" >> "$R/chain.txt"
bash "$D/box/served_ab.sh" served-unit2-final "$M" "$L" "B L L B B L" > "$R/served.out" 2>&1
echo CHAIN6-DONE >> "$R/chain.txt"
