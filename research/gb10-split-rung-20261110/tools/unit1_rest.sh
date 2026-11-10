#!/usr/bin/env bash
# Rig: the rest of lever 1's unit 1 cells on the final head, in order, one lease per cell
# (unit_cell.sh / split_driver.sh yield to landing waiters): Ornith's batch margin arm, the final
# head's full kernel-check and run-spec at both straddle prompts, the deep warm == cold pairs,
# then the served B/T (B = main 42bd4b07e4, T = 9592c2dceb) at 131k, 260k, 32k and 12k.
# No build runs on the unit once the timed cells start: both binaries are waited for first.
# As run: stopped after the full kernel-check; the run-specs and the pairs moved to unit 2
# (unit2_passfail.sh) and the served B/T to bt_unit.sh, one context per unit.
# usage: UNIT1=<unit 1's sparklock name> unit1_rest.sh WAIT_PID
set -u
U=${UNIT1:?unit 1 sparklock name}
R=$HOME/.cache/memra-split-rung; T=$R/wt/research/gb10-split-rung-20261110/tools
H=9592c2dceb; B=42bd4b07e4
L=$R/runs/unit1-rest.log
log() { echo "$(date -Is) $*" >> "$L"; }
waitfor --gone "$1" --timeout 120m --quiet
log "start"
bash "$T/gates_gb10.sh" "$U" head-1a4f769861 src-1a4f769861 am-or-32774-b; log "am-or-b done"
waitfor --ssh spark-$U "grep -q BUILD-DONE ~/.cache/memra-split-rung/logs/build-head-$H.log && grep -q BUILD-DONE ~/.cache/memra-split-rung/logs/build-base-$B.log" --timeout 120m --every 30s --quiet || { log "builds missing"; exit 1; }
ssh spark-$U "cd ~/.cache/memra-split-rung && mkdir -p kt-head-$H/release && for b in bins/head-$H/*; do ln -sf \$HOME/.cache/memra-split-rung/\$b kt-head-$H/release/; done"
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
MOR='$HOME/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf'
bash "$T/unit_cell.sh" "$U" "kc-full-27b-$H" 30 "unset MEMRA_KC_FAST MEMRA_KC_ONLY; bins/head-$H/kernel-check $M27 --require-manifest src-build/tools/kernel-check-27b.cells --require-manifest src-build/tools/kernel-check-step35.cells"; log "kc-full rc=$?"
bash "$T/unit_cell.sh" "$U" "rs-27b-16364-$H" 30 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=\$PWD/prompts/deep-16364.txt bins/head-$H/run-spec $M27"; log "rs-27b rc=$?"
bash "$T/unit_cell.sh" "$U" "rs-or-32748-$H" 35 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=\$PWD/prompts/deep-32748.txt bins/head-$H/run-spec $MOR"; log "rs-or rc=$?"
bash "$T/pairs_gb10.sh" "$U" "head-$H" pairs-27b-deep600 pairs-or-deep1194 pairs-27b-deep664 pairs-or-deep1260 pairs-27b-deep480 pairs-or-deep1101; log "pairs done"
TTL=20 bash "$T/split_driver.sh" "$U" sp131-unit1 "base-$B" "head-$H" 131k; log "131k rc=$?"
TTL=25 bash "$T/split_driver.sh" "$U" sp260-unit1 "base-$B" "head-$H" 260k; log "260k rc=$?"
TTL=15 bash "$T/split_driver.sh" "$U" sp32-unit1 "base-$B" "head-$H" 32k; log "32k rc=$?"
TTL=15 bash "$T/split_driver.sh" "$U" sp12-unit1 "base-$B" "head-$H" 12k; log "12k rc=$?"
log "all done"
