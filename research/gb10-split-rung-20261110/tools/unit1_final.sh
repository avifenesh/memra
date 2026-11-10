#!/usr/bin/env bash
# Rig: unit 1 at the final code head, once its running context (the 260k B/T at 9592c2dceb) has
# ended and the head's binaries are in place: its share of the B/T from the shared queue (B =
# main BASE, T = HEAD), then the full kernel-check at the head (the battery form).
# usage: UNIT1=<unit 1's sparklock name> unit1_final.sh WAIT_PID HEAD BASE "CTX..."
set -u
U=${UNIT1:?unit 1 sparklock name}; P=$1; H=$2; B=$3; CTXS=$4
R=$HOME/.cache/memra-split-rung; T=$R/wt/research/gb10-split-rung-20261110/tools
L=$R/runs/unit1-final.log
log() { echo "$(date -Is) $*" >> "$L"; }
waitfor --gone "$P" --timeout 240m --quiet
bash "$T/bt_queue.sh" "$U" 1 "$B" "$H" "$CTXS" "$R/runs/bins-ready-$H"; log "B/T rc=$?"
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
bash "$T/unit_cell.sh" "$U" "kc-full-27b-$H" 30 "unset MEMRA_KC_FAST MEMRA_KC_ONLY MEMRA_FA_SPLIT; bins/head-$H/kernel-check $M27 --require-manifest src-build/tools/kernel-check-27b.cells --require-manifest src-build/tools/kernel-check-step35.cells"; log "kc-full rc=$?"
log "all done"
