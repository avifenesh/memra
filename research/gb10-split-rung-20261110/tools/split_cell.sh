#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one cell of the deep split rung's served B/T.
#   NN-B / NN-T: a fresh server (B = bins/<BASE>, T = bins/<HEAD>), timestamped log with
#   MEMRA_TICK_TRACE=1, then split_ab.py --set SET. (research/gb10-verify-rows-20261109/tools/
#   rows_cell.sh in this lane's directory.)
# usage: split_cell.sh CAMPAIGN NAME BASE_DIR HEAD_DIR SET
set -u
D=$HOME/.cache/memra-split-rung
CAMP=$1; NAME=$2; B=$D/bins/$3; T=$D/bins/$4; SET=$5
CD=$D/runs/$CAMP; mkdir -p "$CD"
ARM=${NAME#*-}; bin=$B; [ "$ARM" = T ] && bin=$T
OUT=$CD/$NAME; mkdir -p "$OUT"
OUT=$OUT BIN=$bin EXTRA="MEMRA_TICK_TRACE=1" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
spid=$(cat "$OUT/server.pid")
waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1 || { kill -TERM "$spid"; wait $sp; echo "SPLIT-CELL-FAIL $NAME" >> "$CD/campaign.log"; exit 1; }
python3 "$D/box/split_ab.py" "$OUT/ab.jsonl" --set "$SET" --tag "$NAME" >> "$OUT/run.log" 2>&1
kill -TERM "$spid"
waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
wait $sp
grep '^{' "$OUT/ab.jsonl" | cut -c1-260 | sed "s/^/$ARM /" >> "$CD/campaign.log"
grep -c '^{' "$OUT/ab.jsonl" | grep -qx 2 || { echo "SPLIT-CELL-FAIL $NAME (rows missing)" >> "$CD/campaign.log"; exit 1; }
echo "SPLIT-CELL-DONE $NAME" >> "$CD/campaign.log"
