#!/usr/bin/env bash
# Runs ON unit1 (unit 1) under its lease, after unit1_chain.sh: build the final stack (gdn5), the
# kernel-check synthetic arms (GDN-DB-GQA pins the fused K4+K5 against the pair), then the
# interleaved prime A/B K0 = gdn5 with MEMRA_GDN_K45=0 against A = gdn5 defaults.
set -u
D=$HOME/.cache/memra-gb10-next
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
R=$D/runs/unit1-gdn5
mkdir -p "$R"
bash "$D/box/lane_build2.sh" gdn5 "$D/lane5.bundle" > "$R/build.out" 2>&1
L=$(ls -d "$D"/bins/gdn5-* 2>/dev/null | tail -n 1)
echo "gdn5=$L $(tail -n 1 "$R/build.out")" > "$R/chain.txt"
[ -n "$L" ] || { echo CB33-5-BUILD-FAIL >> "$R/chain.txt"; exit 1; }
MEMRA_KC_FAST=1 "$L/kernel-check" "$G8" > "$R/kc-fast.log" 2>&1
echo "kc-fast rc=$? $(tail -n 1 "$R/kc-fast.log")" >> "$R/chain.txt"
grep -E "GDN-DB-GQA" "$R/kc-fast.log" >> "$R/chain.txt"
LANE=$L MAIN=$L bash "$D/box/pp_ab4.sh" "$R/pp33k" 1240 "K0 A A K0"
echo CB33-CHAIN2-DONE >> "$R/chain.txt"
