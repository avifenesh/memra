#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1307): one cell. A fresh server of BIN_DIR
# (serve1307.sh; MODE mtp, or dflash for the client modes dflash, dflash4 and dflash4long), then the
# cell's memory STATE, then client1307.py's requests in MODE, then the server stops.
#   quiet     - nothing (the campaign drops the page cache before the cell);
#   live      - a page-cache fill to MemFree under 3,000 MB with two readers kept going through the
#               requests (live_fill.sh);
#   liveboot  - the same fill and readers started before the server, so they run through the boot
#               too: the boot's spec calibration captures the draft graphs and records the
#               draft-state high-water the requests then pay;
#   bootonly  - nothing, and no requests: the cell is the boot, whose spec calibration opens the
#               first draft-state brackets (the campaign sets the page cache before the cell);
#   shortlate - MemAvailable truly short while a cohort decodes: once the server logs its first
#               cohort, memra#1221's mem_holder.py takes MemAvailable down to TARGET_MB (2,000 by
#               default) with anonymous memory (oom_score_adj 1000, exits by itself after 600 s)
#               and holds it until the requests end. The memory guard is lowered to 1 GiB and the
#               window's kernel log goes to the cell.
# trace.txt keeps the server's [diag1307], [spec], [spec-pool], [admit-trim], [device-trim] and OOM
# lines.
# usage: [TARGET_MB=n] cell1307.sh CAMPAIGN_DIR NAME BIN_DIR MODE STATE
set -u
CD=$1; NAME=$2; BIN=$3; MODE=$4; STATE=$5
H=$(cd "$(dirname "$0")" && pwd)
export TMPDIR=$HOME/.cache/memra-1307/tmp
mkdir -p "$TMPDIR"
OUT=$CD/$NAME; mkdir -p "$OUT"
log() { echo "$(date -Is) $NAME $*" >> "$CD/campaign.log"; }
mi() { grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';'; }
[ -x "$BIN/memra-server" ] || { log "CELL-FAIL no binary $BIN"; exit 1; }
log "start MODE=$MODE STATE=$STATE BIN=${BIN/#$HOME/~}"
case $MODE in mtp) SMODE=mtp ;; dflash*) SMODE=dflash ;; *) log "CELL-FAIL mode $MODE"; exit 2 ;; esac
STOP=$OUT/stop
t0=$(date '+%F %T')
if [ "$STATE" = liveboot ]; then
  bash "$H/live_fill.sh" 3000 2 1200 "$STOP" "$OUT/memfree.log" >> "$OUT/run.log" 2>&1
fi
GUARD=3145728; [ "$STATE" = shortlate ] && GUARD=1048576
OUT=$OUT BIN=$BIN MODE=$SMODE GUARD_KB=$GUARD bash "$H/serve1307.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
spid=$(cat "$OUT/server.pid")
waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" \
  --every 5s --timeout 15m >> "$OUT/run.log" 2>&1 || { touch "$STOP"; kill -TERM "$spid"; wait $sp; log "CELL-FAIL not ready"; exit 1; }
if [ "$STATE" = live ]; then
  bash "$H/live_fill.sh" 3000 2 900 "$STOP" "$OUT/memfree.log" >> "$OUT/run.log" 2>&1
fi
echo "before requests: $(mi)" >> "$OUT/run.log"
if [ "$STATE" = shortlate ]; then
  python3 "$H/client1307.py" "$OUT/rows.jsonl" "$MODE" "$NAME" >> "$OUT/run.log" 2>&1 &
  cp=$!
  waitfor --until "grep -q 'cohort engaged' $OUT/server.log" --fail-if "! kill -0 $cp" \
    --every 1s --timeout 3m >> "$OUT/run.log" 2>&1 || log "no cohort before the holder"
  echo "$(date -Is) cohort engaged, holder on: $(mi)" >> "$OUT/run.log"
  python3 "$H/../../spec-capture-headroom-20261112/tools/mem_holder.py" "${TARGET_MB:-2000}" 600 > "$OUT/holder.log" 2>&1 &
  hp=$!
  waitfor --gone "$cp" --timeout 12m >> "$OUT/run.log" 2>&1 || kill -TERM "$cp"
  wait "$cp"
  echo "$(date -Is) after the requests: $(mi)" >> "$OUT/run.log"
  kill -TERM "$hp"; waitfor --gone "$hp" --timeout 1m >> "$OUT/run.log" 2>&1 || kill -KILL "$hp"
  wait "$hp"
elif [ "$STATE" != bootonly ]; then
  python3 "$H/client1307.py" "$OUT/rows.jsonl" "$MODE" "$NAME" >> "$OUT/run.log" 2>&1
fi
touch "$STOP"
kill -TERM "$spid"
waitfor --gone "$spid" --timeout 5m >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
wait $sp
if [ "$STATE" = shortlate ]; then
  journalctl -k --since "$t0" --no-pager > "$OUT/kernel.log" 2>&1
  grep -ciE 'out of memory|oom-kill|oom_reaper|invoked oom-killer' "$OUT/kernel.log" > "$OUT/kernel-oom-count.txt"
fi
grep -E '\[diag1307\]|\[spec\]|\[spec-pool\]|\[admit-trim\]|\[device-trim\]|out of memory|OUT_OF_MEMORY|panicked' \
  "$OUT/server.log" > "$OUT/trace.txt"
n=$(grep -c '"status": 200' "$OUT/rows.jsonl" 2>/dev/null)
log "done rows_200=${n:-0} empty_text=$(grep -c 'empty text' "$OUT/rows.jsonl" 2>/dev/null) oom_lines=$(grep -c -E 'out of memory|OUT_OF_MEMORY' "$OUT/server.log")"
