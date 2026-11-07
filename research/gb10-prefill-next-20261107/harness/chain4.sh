#!/usr/bin/env bash
# Runs ON unit2 under the lease: build the stacked lane head (gate/up silu launch, GDN out-norm
# seed), kernel-check synthetic arms plus the model's F8F4 v2 cells, then the interleaved prime
# A/B: M = the PR-1 binary (gdn3), GU0 = the stacked binary with MEMRA_FFN_GATEUP_FUSE=0 (adds the
# out-norm seed), A = the stacked binary's defaults.
set -u
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
P1=$(ls -d "$D"/bins/gdn3-* | tail -n 1)
R=$D/runs/gdn4
mkdir -p "$R"
bash "$D/box/lane_build.sh" gdn4 > "$R/build.out" 2>&1
L=$(ls -d "$D"/bins/gdn4-* 2>/dev/null | tail -n 1)
echo "build: $(tail -n 1 "$R/build.out") bin=$L base=$P1" > "$R/chain.txt"
[ -n "$L" ] || { echo CHAIN4-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GATED-RMSNORM-E4M3V2|ADD-RMS-NORM-E4M3V2|NVFP4-DP4A-RP-T4|GDN-DB-GQA" "$R/kc-fast.log" >> "$R/chain.txt"
MEMRA_KC_ONLY=f8f4-v2 "$L/kernel-check" "$G8" > "$R/kc-f8f4.log" 2>&1
echo "kc-f8f4 rc=$? $(tail -n 1 "$R/kc-f8f4.log")" >> "$R/chain.txt"
grep -E "MMQ-F8F4-V2-GATEUP|MMQ-F8F4-V2 SUMMARY" "$R/kc-f8f4.log" >> "$R/chain.txt"
LANE=$L MAIN=$P1 bash "$D/box/pp_ab3.sh" "$R/pp33k" 1240 "M GU0 A A GU0 M"
echo CHAIN4-DONE >> "$R/chain.txt"
