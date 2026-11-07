#!/usr/bin/env bash
# Runs ON unit1 (unit 1) under its lease: lane tree + warm target, the PR-1 binary (gdn3) and the
# stacked binary (gdn4) from their bundles, kernel-check synthetic arms and the model's F8F4 v2
# cells on gdn4, then the interleaved prime A/B: M = gdn3, GU0 = gdn4 with
# MEMRA_FFN_GATEUP_FUSE=0 (adds the GDN out-norm seed), A = gdn4 defaults.
set -u
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
R=$D/runs/unit1-gdn4
mkdir -p "$R" "$D/logs"
bash "$D/box/setup_tree.sh" "$HOME/projects/memra-cells-repo" "$HOME/.cache/memra-base-target" > "$R/setup.log" 2>&1
bash "$D/box/lane_build2.sh" gdn3 "$D/lane3.bundle" > "$R/build3.out" 2>&1
bash "$D/box/lane_build2.sh" gdn4 "$D/lane4.bundle" > "$R/build4.out" 2>&1
P1=$(ls -d "$D"/bins/gdn3-* 2>/dev/null | tail -n 1)
L=$(ls -d "$D"/bins/gdn4-* 2>/dev/null | tail -n 1)
echo "gdn3=$P1 gdn4=$L $(tail -n 1 "$R/build3.out") / $(tail -n 1 "$R/build4.out")" > "$R/chain.txt"
[ -n "$P1" ] && [ -n "$L" ] || { echo CB33-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GATED-RMSNORM-E4M3V2|ADD-RMS-NORM-E4M3V2|NVFP4-DP4A-RP-T4|GDN-DB-GQA" "$R/kc-fast.log" >> "$R/chain.txt"
MEMRA_KC_ONLY=f8f4-v2 "$L/kernel-check" "$G8" > "$R/kc-f8f4.log" 2>&1
echo "kc-f8f4 rc=$? $(tail -n 1 "$R/kc-f8f4.log")" >> "$R/chain.txt"
grep -E "MMQ-F8F4-V2-GATEUP|MMQ-F8F4-V2 SUMMARY" "$R/kc-f8f4.log" >> "$R/chain.txt"
LANE=$L MAIN=$P1 bash "$D/box/pp_ab3.sh" "$R/pp33k" 1240 "M GU0 A A GU0 M"
echo CB33-CHAIN-DONE >> "$R/chain.txt"
