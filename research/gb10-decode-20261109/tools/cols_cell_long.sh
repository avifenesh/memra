#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: one cell of round 5 lever 3a served A/B with the 131k request.
#   NN-B / NN-T: a fresh server (B = bins/eosb-<base>, T = bins/eost-<head>), timestamped log with
#                MEMRA_TICK_TRACE=1, then eos_ab.py --long (g1k, g1k-t2, s1k, g131k)
#   gate:        the session-gate oracle on the T build (engine unchanged by the lever; K=3, 48 new
#                tokens per turn)
# usage: eos_cell.sh CAMPAIGN NAME BASE10 HEAD10
set -u
D=$HOME/.cache/memra-gb10-r4
CAMP=$1; NAME=$2; B=$D/bins/eosb-$3; T=$D/bins/colst-$4
CD=$D/runs/$CAMP; mkdir -p "$CD"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
case "$NAME" in
  gate)
    (MEMRA_SPEC_K=3 MEMRA_NGEN=48 "$T/session-gate" "$G8") > "$CD/session-gate.log" 2>&1
    echo "session-gate rc=$? $(tail -n 1 "$CD/session-gate.log")" >> "$CD/campaign.log"
    ;;
  *)
    ARM=${NAME#*-}; bin=$B; [ "$ARM" = T ] && bin=$T
    OUT=$CD/$NAME; mkdir -p "$OUT"
    OUT=$OUT BIN=$bin EXTRA="MEMRA_TICK_TRACE=1" WRAP="python3 $D/box/tsrun.py" bash "$D/box/serve.sh" &
    sp=$!
    sleep 2
    spid=$(cat "$OUT/server.pid")
    waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" --every 5s --timeout 15m > "$OUT/run.log" 2>&1 || { kill -TERM "$spid"; wait $sp; echo "EOS-CELL-FAIL $NAME" >> "$CD/campaign.log"; exit 1; }
    python3 "$D/box/eos_ab.py" "$OUT/ab.jsonl" --tag "$NAME" --long >> "$OUT/run.log" 2>&1
    kill -TERM "$spid"
    waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
    wait $sp
    grep '^{' "$OUT/ab.jsonl" | cut -c1-260 | sed "s/^/$ARM /" >> "$CD/campaign.log"
    ;;
esac
echo "EOS-CELL-DONE $NAME" >> "$CD/campaign.log"
