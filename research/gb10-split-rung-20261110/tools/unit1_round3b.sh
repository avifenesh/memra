#!/usr/bin/env bash
# Rig: the review's third round at the final code head on GB10 unit 1, without the kernel-check
# (its binary is byte-identical to the one that ran ALL GREEN at bf25fcb629): run-spec K=1..8 at
# both straddle prompts and the six warm == cold pair cases (cells suffixed -HEAD). One lease
# per cell; other lanes' and landing tickets go first at every cell boundary.
# usage: UNIT1=<unit 1's sparklock name> unit1_round3b.sh HEAD
set -u
U=${UNIT1:?unit 1 sparklock name}; H=$1
R=$HOME/.cache/memra-split-rung; T=$R/wt/research/gb10-split-rung-20261110/tools
L=$R/runs/unit1-round3b.log
log() { echo "$(date -Is) $*" >> "$L"; }
log "start"
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
MOR='$HOME/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf'
cell() { bash "$T/unit_cell.sh" "$U" "$1" "$2" "$3"; log "$1 rc=$?"; }
cell "rs-27b-16364-$H" 30 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=\$PWD/prompts/deep-16364.txt bins/head-$H/run-spec $M27"
cell "rs-or-32748-$H" 35 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=\$PWD/prompts/deep-32748.txt bins/head-$H/run-spec $MOR"
SUFFIX="-$H" bash "$T/pairs_gb10.sh" "$U" "head-$H"; log "pairs done"
log "all done"
