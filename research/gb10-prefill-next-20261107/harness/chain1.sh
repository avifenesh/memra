#!/usr/bin/env bash
# Runs ON unit2 under the lease: kernel-check synthetic arms on the lane binary, the GDN K4/K5
# probe, then the interleaved prime A/B on a 33k record prompt.
set -u
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
L=$D/bins/gdn1-48265176f0
M=$D/bins/main-9805016bb1
R=$D/runs/gdn1
mkdir -p "$R"
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" > "$R/chain.txt"
grep -E "GDN-DB-GQA|gdn_chunked" "$R/kc-fast.log" >> "$R/chain.txt"
bash "$D/probe/run_probe_gdn.sh" "$D/tree" "$R/probe-gdn.log" 4096 8192
echo "probe done" >> "$R/chain.txt"
LANE=$L MAIN=$M bash "$D/box/pp_ab.sh" "$R/pp33k" 1240 "M B D W F F W D B M"
echo CHAIN1-DONE >> "$R/chain.txt"
