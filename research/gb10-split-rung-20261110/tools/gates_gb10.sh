#!/usr/bin/env bash
# Rig: the deep split rung's gate cells on one GB10 unit, one sparklock lease per cell
# (unit_cell.sh). Each cell reaches the new rung: its prompt sits on both sides of a floor
# (16,384 keys for the 27B's 4 KV heads, 32,768 for Ornith's 2) or past it. A failed cell is a
# result and does not stop the driver.
# usage: gates_gb10.sh UNIT HEAD_BIN_DIR SRC_DIR [CELL...]   (no CELL: every cell, in order)
set -u
UNIT=$1; HB=$2; SRC=$3; shift 3
T=$(cd "$(dirname "$0")" && pwd)
K="kt-$HB/release"
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
MOR='$HOME/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf'
P='$PWD/prompts'
AM="CARGO_TARGET_DIR=\$PWD/kt-$HB bash $SRC/tools/argmax-margin-gate.sh"
declare -A TTL CMD
add() { TTL[$1]=$2; CMD[$1]=$3; ORDER+=("$1"); }
ORDER=()
add kc-full-27b 30 "unset MEMRA_KC_FAST MEMRA_KC_ONLY; $K/kernel-check $M27 --require-manifest $SRC/tools/kernel-check-27b.cells --require-manifest $SRC/tools/kernel-check-step35.cells"
add rs-27b-16364 30 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=$P/deep-16364.txt $K/run-spec $M27"
add ssg-27b-16364 25 "MEMRA_PROMPT_FILE=$P/deep-16364.txt $K/spec-serve-gate $M27 --steps 8 --ts 1,2,4 --spec-k 3"
add rs-or-32748 35 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=64 MEMRA_PROMPT_FILE=$P/deep-32748.txt $K/run-spec $MOR"
add ssg-or-32748 25 "MEMRA_PROMPT_FILE=$P/deep-32748.txt $K/spec-serve-gate $MOR --steps 8 --ts 1,2,4 --spec-k 3"
add am-27b-16390-e 45 "$AM $M27 --prompt $P/deep-16390.txt --logdir \$PWD/cells/am-27b-16390-e/gate"
add am-27b-16390-b 45 "$AM $M27 --prompt $P/deep-16390.txt --decode batch --logdir \$PWD/cells/am-27b-16390-b/gate"
add am-or-32774-e 45 "$AM $MOR --prompt $P/deep-32774.txt --logdir \$PWD/cells/am-or-32774-e/gate"
add am-or-32774-b 45 "$AM $MOR --prompt $P/deep-32774.txt --decode batch --logdir \$PWD/cells/am-or-32774-b/gate"
add rs-27b-65500 35 "MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 MEMRA_PROMPT_FILE=$P/deep-65500.txt $K/run-spec $M27"
add am-27b-p4-16k-e 55 "$AM $M27 --prompt $P/p4-16k.txt --logdir \$PWD/cells/am-27b-p4-16k-e/gate"
add am-27b-p4-16k-b 55 "$AM $M27 --prompt $P/p4-16k.txt --decode batch --logdir \$PWD/cells/am-27b-p4-16k-b/gate"
add gd-27b-16300 70 "$K/graph-decode-gate $M27 16300 160"
add gd-or-32700 60 "$K/graph-decode-gate $MOR 32700 160"
L=$HOME/.cache/memra-split-rung/runs/gates-$UNIT.log
cells=("$@"); [ ${#cells[@]} -gt 0 ] || cells=("${ORDER[@]}")
for c in "${cells[@]}"; do
  [ -n "${CMD[$c]:-}" ] || { echo "$(date -Is) unknown cell $c" >> "$L"; continue; }
  echo "$(date -Is) $c start" >> "$L"
  bash "$T/unit_cell.sh" "$UNIT" "$c" "${TTL[$c]}" "${CMD[$c]}"
  echo "$(date -Is) $c rc=$?" >> "$L"
done
echo "$(date -Is) all gate cells done" >> "$L"
