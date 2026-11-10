#!/usr/bin/env bash
# Rig (usage: BUILD_HOST=<sm_120a build host> final_5090.sh HEAD): the RTX 5090's cells at a head (its ladder is today's): one untimed kernel-check
# (synthetic sections, rows to 262k) and run-spec K=1..8 on Qwen3.8-27B in the battery's form.
set -u
BH=${BUILD_HOST:?the sm_120a build host}
R=$HOME/.cache/memra-split-rung; H=${1:?head commit}
O=$R/runs/final-5090-$H; mkdir -p "$O"
waitfor --ssh "$BH" "grep -q BUILD-DONE ~/.cache/memra-split-rung/logs/build-head-$H.log" --timeout 90m --every 30s --quiet || exit 1
mkdir -p "$R/bins/head-$H-120a"
scp -q "$BH:.cache/memra-split-rung/bins/head-$H/*" "$R/bins/head-$H-120a/"
sha256sum "$R/bins/head-$H-120a/"* > "$O/sha256.txt"
gpulock run --ttl 15m --wait 300m --for "memra deep split rung: kernel-check at the final head $H, the 5090 on today's ladder (untimed pass/fail)" -- \
  env MEMRA_KC_FAST=1 MEMRA_KC_ROWS_DEPTH=262144 timeout 12m "$R/bins/head-$H-120a/kernel-check" > "$O/kc.log" 2>&1
gpulock run --ttl 20m --wait 300m --for "memra deep split rung: run-spec K=1..8 on the 27B at the final head $H (untimed pass/fail)" -- \
  env MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 timeout 15m "$R/bins/head-$H-120a/run-spec" "$HOME/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" > "$O/rs.log" 2>&1
echo done >> "$O/sha256.txt"
