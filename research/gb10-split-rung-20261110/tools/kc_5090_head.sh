#!/usr/bin/env bash
# Rig: the RTX 5090's FAST kernel-check at a head (untimed pass/fail; the card's power mode at
# the time is recorded from the lease's start), binaries copied from the sm_120a build host.
# usage: BUILD_HOST=<sm_120a build host> kc_5090_head.sh HEAD
set -u
BH=${BUILD_HOST:?the sm_120a build host}; H=${1:?head commit}
R=$HOME/.cache/memra-split-rung; O=$R/runs/final-5090-$H; mkdir -p "$O"
waitfor --ssh "$BH" "grep -q BUILD-DONE ~/.cache/memra-split-rung/logs/build-head-$H.log" --timeout 60m --every 30s --quiet || exit 1
mkdir -p "$R/bins/head-$H-120a"
scp -q "$BH:.cache/memra-split-rung/bins/head-$H/*" "$R/bins/head-$H-120a/"
sha256sum "$R/bins/head-$H-120a/"* > "$O/sha256.txt"
gpulock run --ttl 15m --wait 300m --for "memra deep split rung: FAST kernel-check at $H, the 5090 on today's ladder (untimed pass/fail)" -- \
  env MEMRA_KC_FAST=1 MEMRA_KC_ROWS_DEPTH=262144 timeout 12m "$R/bins/head-$H-120a/kernel-check" > "$O/kc.log" 2>&1
echo "done $(date -Is)" >> "$O/sha256.txt"
