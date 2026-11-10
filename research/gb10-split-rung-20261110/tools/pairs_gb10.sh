#!/usr/bin/env bash
# Rig: warm == cold pair cells at depth on one GB10 unit (pair_cells_deep.py), one case (a warm
# three-turn boot and two cold boots) per sparklock lease (unit_cell.sh). Greedy, 64 tokens a
# turn, the served route (the 27B with its masked MTP draft, Ornith with its own MTP head), the
# grid path expected. Turn 1 sits below a split floor, straddles it (the turns' decode crosses
# it), or sits past it: 16,384 for the 27B (4 KV heads), 32,768 for Ornith (2).
# usage: [SUFFIX=-HEAD] pairs_gb10.sh UNIT HEAD_BIN_DIR [CASE...]  (SUFFIX names the cells, so a
# rerun at another head keeps the first run's cells)
set -u
UNIT=$1; HB=$2; shift 2
S=${SUFFIX:-}
T=$(cd "$(dirname "$0")" && pwd)
M27='$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf'
DR27='MEMRA_MTP_DRAFT=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf'
MOR='$HOME/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf'
BIN="\$HOME/.cache/memra-split-rung/bins/$HB/memra-server"
declare -A CMD
ORDER=()
add() { CMD[$1]=$2; ORDER+=("$1"); }
pair() {  # name model shape env
  add "$1" "cd box && CELL_PORT=18193 PAIR_MODEL=$2 timeout 28m python3 pair_cells_deep.py \$HOME/.cache/memra-split-rung/cells/$1$S/out \"$1$S=$BIN|$3|64,64,64|MEMRA_CTX=40960 MEMRA_PREFIX_STABLE_BOUNDARY=1 $4 _EXPECT=grid\""
}
pair pairs-27b-deep480 "$M27" chat-deep480 "$DR27"
pair pairs-27b-deep600 "$M27" chat-deep600 "$DR27"
pair pairs-27b-deep664 "$M27" chat-deep664 "$DR27"
pair pairs-or-deep1101 "$MOR" chat-deep1101 ""
pair pairs-or-deep1194 "$MOR" chat-deep1194 ""
pair pairs-or-deep1260 "$MOR" chat-deep1260 ""
L=$HOME/.cache/memra-split-rung/runs/pairs-$UNIT.log
cases=("$@"); [ ${#cases[@]} -gt 0 ] || cases=("${ORDER[@]}")
for c in "${cases[@]}"; do
  [ -n "${CMD[$c]:-}" ] || { echo "$(date -Is) unknown case $c" >> "$L"; continue; }
  echo "$(date -Is) $c start" >> "$L"
  bash "$T/unit_cell.sh" "$UNIT" "$c$S" 30 "${CMD[$c]}"
  echo "$(date -Is) $c rc=$?" >> "$L"
done
echo "$(date -Is) all pair cases done" >> "$L"
