#!/usr/bin/env bash
# Runs ON unit2 under the lease: build the glue-2 head (attention out-gate seed, GDN conv + L2),
# kernel-check synthetic arms, then the prime A/B M = the #1089 binary (gdn6) vs A = glue-2.
set -u
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
P=$(ls -d "$D"/bins/gdn6-* | tail -n 1)
R=$D/runs/glue2
mkdir -p "$R"
bash "$D/box/lane_build2.sh" glue2 "$D/lane7.bundle" > "$R/build.out" 2>&1
L=$(ls -d "$D"/bins/glue2-* 2>/dev/null | tail -n 1)
echo "base=$P lane=$L $(tail -n 1 "$R/build.out")" > "$R/chain.txt"
[ -n "$L" ] || { echo CHAIN7-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GDN-CONV-L2|SIG-MUL-E4M3V2|GDN-DB-GQA|GATED-RMSNORM" "$R/kc-fast.log" >> "$R/chain.txt"
LANE=$L MAIN=$P bash "$D/box/pp_ab4.sh" "$R/pp33k" 1240 "M A A M"
echo CHAIN7-DONE >> "$R/chain.txt"
