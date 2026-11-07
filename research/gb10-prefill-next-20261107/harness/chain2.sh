#!/usr/bin/env bash
# Runs ON unit2 under the lease: build the norm-fusion arm, run the GDN K4/K5 probe, kernel-check
# synthetic arms, then the interleaved prime A/B over the norm-fusion levels on a 33k prompt.
set -u
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
M=$D/bins/main-9805016bb1
R=$D/runs/gdn2
mkdir -p "$R"
bash "$D/box/lane_build.sh" gdn2 > "$R/build.out" 2>&1
L=$(ls -d "$D"/bins/gdn2-* 2>/dev/null | tail -n 1)
echo "build: $(tail -n 1 "$R/build.out") bin=$L" > "$R/chain.txt"
[ -n "$L" ] || { echo CHAIN2-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
bash "$D/probe/run_probe_gdn.sh" "$D/tree" "$R/probe-gdn.log" 4096 8192
echo "probe done" >> "$R/chain.txt"
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GDN-DB-GQA|ADD-RMS-NORM-E4M3V2|add_rms_norm fused|NVFP4-DP4A-RP-T4" "$R/kc-fast.log" >> "$R/chain.txt"
LANE=$L MAIN=$M bash "$D/box/pp_ab2.sh" "$R/pp33k" 1240 "Z G G1 G3 GQ A A GQ G3 G1 G Z"
echo CHAIN2-DONE >> "$R/chain.txt"
