#!/usr/bin/env bash
# Rig: the candidate's margin and spec-serve-gate cells again at the final head 9592c2dceb on
# GB10 unit 2 (they first ran on unit 1 at 1a4f769861), so the exactness tables stand on the
# head under review: the 27B at 16,390 (eager and batch) and on real text at 16,390, Ornith at
# 32,774 (eager and batch), spec-serve-gate at both straddle prompts. Same commands, same
# prompts; one sparklock lease per cell, yielding to any waiter that is not ours.
# usage: UNIT2=<unit 2's sparklock name> unit2_final_gates.sh WAIT_PID
set -u
U=${UNIT2:?unit 2 sparklock name}
R=$HOME/.cache/memra-split-rung; T=$R/wt/research/gb10-split-rung-20261110/tools
H=9592c2dceb
L=$R/runs/unit2-final-gates.log
log() { echo "$(date -Is) $*" >> "$L"; }
export YIELD_ANY=1
waitfor --gone "$1" --timeout 240m --quiet
log "start"
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
MOR='$HOME/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf'
G="CARGO_TARGET_DIR=\$PWD/kt-head-$H bash src-1a4f769861/tools/argmax-margin-gate.sh"
cell() { bash "$T/unit_cell.sh" "$U" "$1" "$2" "$3"; log "$1 rc=$?"; }
cell "am-27b-16390-e-$H" 30 "$G $M27 --prompt \$PWD/prompts/deep-16390.txt --logdir \$PWD/cells/am-27b-16390-e-$H/gate"
cell "am-27b-16390-b-$H" 30 "$G $M27 --prompt \$PWD/prompts/deep-16390.txt --decode batch --logdir \$PWD/cells/am-27b-16390-b-$H/gate"
cell "am-27b-p4-16390-e-$H" 30 "$G $M27 --prompt \$PWD/prompts/p4-16390.txt --logdir \$PWD/cells/am-27b-p4-16390-e-$H/gate"
cell "am-or-32774-e-$H" 40 "$G $MOR --prompt \$PWD/prompts/deep-32774.txt --logdir \$PWD/cells/am-or-32774-e-$H/gate"
cell "am-or-32774-b-$H" 40 "$G $MOR --prompt \$PWD/prompts/deep-32774.txt --decode batch --logdir \$PWD/cells/am-or-32774-b-$H/gate"
cell "ssg-27b-16364-$H" 30 "MEMRA_PROMPT_FILE=\$PWD/prompts/deep-16364.txt kt-head-$H/release/spec-serve-gate $M27 --steps 8 --ts 1,2,4 --spec-k 3"
cell "ssg-or-32748-$H" 35 "MEMRA_PROMPT_FILE=\$PWD/prompts/deep-32748.txt kt-head-$H/release/spec-serve-gate $MOR --steps 8 --ts 1,2,4 --spec-k 3"
log "all done"
