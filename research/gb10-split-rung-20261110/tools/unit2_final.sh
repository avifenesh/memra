#!/usr/bin/env bash
# Rig: unit 2 at the final code head, once its earlier chain has ended and the head's binaries
# are in place: the six warm == cold pair cases again (cells suffixed -HEAD), run-spec K=1..8 at
# both straddle prompts, the argmax margin gate on a third model (Qwen3.5-9B, 16 query heads over
# 4 KV heads, so its floor is 16,384) on the real-text prompt that straddles it, eager and batch,
# then its share of the B/T from the shared queue. One lease per cell, yielding to any waiter
# that is not one of ours (YIELD_ANY).
# usage: UNIT2=<unit 2's sparklock name> unit2_final.sh WAIT_PID HEAD BASE "CTX..."
set -u
U=${UNIT2:?unit 2 sparklock name}; P=$1; H=$2; B=$3; CTXS=$4
R=$HOME/.cache/memra-split-rung; T=$R/wt/research/gb10-split-rung-20261110/tools
L=$R/runs/unit2-final.log
log() { echo "$(date -Is) $*" >> "$L"; }
export YIELD_ANY=1
waitfor --gone "$P" --timeout 240m --quiet
waitfor --file "$R/runs/bins-ready-$H" --timeout 240m --quiet || { log "no binaries"; exit 1; }
log "start"
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
MOR='$HOME/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf'
M9='$HOME/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf'
G="CARGO_TARGET_DIR=\$PWD/kt-head-$H bash src-1a4f769861/tools/argmax-margin-gate.sh"
cell() { bash "$T/unit_cell.sh" "$U" "$1" "$2" "$3"; log "$1 rc=$?"; }
SUFFIX="-$H" bash "$T/pairs_gb10.sh" "$U" "head-$H"; log "pairs done"
cell "rs-27b-16364-$H" 30 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=\$PWD/prompts/deep-16364.txt bins/head-$H/run-spec $M27"
cell "rs-or-32748-$H" 35 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=\$PWD/prompts/deep-32748.txt bins/head-$H/run-spec $MOR"
cell "am-9b-p4-16390-e-$H" 30 "$G $M9 --prompt \$PWD/prompts/p4-16390.txt --logdir \$PWD/cells/am-9b-p4-16390-e-$H/gate"
cell "am-9b-p4-16390-b-$H" 30 "$G $M9 --prompt \$PWD/prompts/p4-16390.txt --decode batch --logdir \$PWD/cells/am-9b-p4-16390-b-$H/gate"
bash "$T/bt_queue.sh" "$U" 2 "$B" "$H" "$CTXS"; log "B/T rc=$?"
log "all done"
