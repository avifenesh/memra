#!/usr/bin/env bash
# memra#1257 variant: the cold request goes first (specpub_ab_first.py). Runs ON a GB10 host under the caller's lease: one cell of the spec boundary publication's served
# B/T. NN-<ARM>: a fresh server of that arm's binary (B = bins/<BASE>, T = bins/<HEAD>, U =
# bins/<U_DIR>, any other letter from an ARM=DIR argument) as the launcher serves Qwen3.8-27B
# (serve.sh: the MTP speculative path, 262,144 context), every server line timestamped with
# MEMRA_TICK_TRACE=1 (tsrun.py), then specpub_ab.py --set SET, then the server stops.
# usage: specpub_cell.sh CAMPAIGN NAME BASE_DIR HEAD_DIR SET [U_DIR [ARM=DIR...]]
set -u
D=$HOME/.cache/memra-1221
CAMP=$1; NAME=$2; B=$D/bins/$3; T=$D/bins/$4; SET=$5; U=$D/bins/${6:-none}
CD=$D/runs/$CAMP; mkdir -p "$CD"
ARM=${NAME#*-}; bin=$D/bins/none
case $ARM in B) bin=$B ;; T) bin=$T ;; U) bin=$U ;; esac
[ $# -gt 6 ] && { shift 6; for kv in "$@"; do [ "${kv%%=*}" = "$ARM" ] && bin=$D/bins/${kv#*=}; done; }
[ -x "$bin/memra-server" ] || { echo "SPECPUB-CELL-FAIL $NAME (no binary for arm $ARM)" >> "$CD/campaign.log"; exit 1; }
OUT=$CD/$NAME; mkdir -p "$OUT"
OUT=$OUT BIN=$bin EXTRA="MEMRA_TICK_TRACE=1" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
spid=$(cat "$OUT/server.pid")
waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1 || { kill -TERM "$spid"; wait $sp; echo "SPECPUB-CELL-FAIL $NAME" >> "$CD/campaign.log"; exit 1; }
# MID: a command run once the server is ready, before the requests (fill_cache.sh: low free pages
# at request time, the condition in which the pre-capture pool trim fires).
[ -n "${MID:-}" ] && $MID >> "$OUT/run.log" 2>&1
python3 "$D/box/specpub_ab_first.py" "$OUT/ab.jsonl" --set "$SET" --tag "$NAME" >> "$OUT/run.log" 2>&1
kill -TERM "$spid"
waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
wait $sp
grep '^{' "$OUT/ab.jsonl" | cut -c1-260 | sed "s/^/$ARM /" >> "$CD/campaign.log"
grep -c '^{' "$OUT/ab.jsonl" | grep -qx 2 || { echo "SPECPUB-CELL-FAIL $NAME (rows missing)" >> "$CD/campaign.log"; exit 1; }
echo "SPECPUB-CELL-DONE $NAME" >> "$CD/campaign.log"
