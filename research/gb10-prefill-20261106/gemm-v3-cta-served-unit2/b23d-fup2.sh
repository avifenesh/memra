#!/usr/bin/env bash
# Runs ON b23d under the caller's lease: the rest of the paused served campaign (cell 1, B, ran
# 10:54-11:05 in runs/fup-campaign/served), cells L N N L B, then one merged directory with the
# six cells in campaign order and its summary, then the follow-up's 16k argmax gate.
# usage: b23d-fup2.sh LANDED_BIN FOLLOWUP_BIN TREE
set -u
LB=$1; NB=$2; TREE=$3
D=$HOME/.cache/memra-f8f4v3
R=$D/runs/fup-campaign
export TMPDIR=$D/tmp PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
log(){ echo "$(date +%T) $*"; }
P2=$R/served2; mkdir -p "$P2"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader > "$P2/clocks-start.txt"
bash "$D/h/campaign.sh" "$P2" "32768 131072 260000" "1 1 1" \
  "L|$LB|" "N|$NB|" "N|$NB|" "L|$LB|" "B|$LB|MEMRA_MMQ_F8F4_V3=0" > "$P2/campaign.out" 2>&1
nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader > "$P2/clocks-end.txt"
M=$R/merged; mkdir -p "$M"
cp -a "$R/served/01-B" "$M/01-B"
n=1
for c in 01-L 02-N 03-N 04-L 05-B; do n=$((n+1)); cp -a "$P2/$c" "$M/$(printf %02d $n)-${c#*-}"; done
python3 "$D/h/summarize.py" "$M" > "$M/summary.txt" 2>&1
log "served done"; head -n 30 "$M/summary.txt"
bash "$D/argmax-fup.sh" "$NB" "$TREE" "$R/argmax" > "$R/argmax.out" 2>&1
tail -n 2 "$R/argmax.out"
echo B23D-FUP2-DONE
