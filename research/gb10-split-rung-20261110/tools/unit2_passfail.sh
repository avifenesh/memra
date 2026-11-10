#!/usr/bin/env bash
# Rig (usage: UNIT2=<unit 2's sparklock name> unit2_passfail.sh): lever 1's pass/fail cells at the final head on GB10 unit 2, in parallel with unit 1's
# margin and served cells: run-spec K=1..8 (STEP-IDENTITY included) at both straddle prompts,
# then the six deep warm == cold pair cases. One sparklock lease per cell; the unit's other lane
# has first call, so every cell boundary yields to any waiter that is not one of ours (YIELD_ANY).
# The binaries are unit 1's sm_121a build of the head, copied with their sha256 checked.
set -u
U=${UNIT2:?unit 2 sparklock name}
R=$HOME/.cache/memra-split-rung; T=$R/wt/research/gb10-split-rung-20261110/tools
H=9592c2dceb
L=$R/runs/unit2-passfail.log
log() { echo "$(date -Is) $*" >> "$L"; }
export YIELD_ANY=1
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
MOR='$HOME/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf'
log "start"
bash "$T/unit_cell.sh" "$U" "rs-27b-16364-$H" 30 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=\$PWD/prompts/deep-16364.txt bins/head-$H/run-spec $M27"; log "rs-27b rc=$?"
bash "$T/unit_cell.sh" "$U" "rs-or-32748-$H" 35 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=\$PWD/prompts/deep-32748.txt bins/head-$H/run-spec $MOR"; log "rs-or rc=$?"
bash "$T/pairs_gb10.sh" "$U" "head-$H" pairs-27b-deep600 pairs-or-deep1194 pairs-27b-deep664 pairs-or-deep1260 pairs-27b-deep480 pairs-or-deep1101; log "pairs done"
log "all done"
