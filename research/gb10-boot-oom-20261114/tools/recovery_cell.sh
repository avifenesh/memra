#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1330): the state of transition_cell.sh
# (memra#1316) with driver-level creations instead of server boots, every failed creation retried
# with no cap at its arm's cadence. Written to ~/.cache/memra-1330/runs/NAME/.
# The page cache is filled to MemFree < 3000 MB and READERS readers kept running (fill_live.sh).
# Then rounds, until UNTIL (a unix time), ROUNDS rounds or a STOP file:
#   a holder backs HOLD_GIB of device memory, keeps it HOLD_S seconds and exits cleanly
#   (ctx_probe.py hold, its own creation retried at the round's first arm and kept in
#   holders.jsonl);
#   the round waits (ARM_WAIT_S at most) until MemFree is back under ARM_MB;
#   K creations follow (ctx_probe.py round: each `recover ARM` in a fresh process), GAP_MS apart,
#   the arms of ARMS in turn; the first arm alternates between rounds.
# When MemFree does not come back under ARM_MB the round makes no creation and the episode ends:
# the readers stop for PAUSE_S seconds and start again, then SETTLE_S seconds with no creation.
# creations.jsonl has one row per creation with every try; memwatch.jsonl MemFree, MemAvailable,
# Cached and the large free blocks every 50 ms; nvrm.log the kernel's NVRM lines. Reads only;
# changes no setting. Stop: touch runs/NAME/STOP (ends after the round in progress).
# usage: NAME=x READERS=n K=n GAP_MS=n ARMS="fixed50 ladder" HOLD_GIB=n ARM_MB=n [HOLD_S=2] [ARM_WAIT_S=45]
#        [PAUSE_S=60] [SETTLE_S=20] [ROUNDS=n] [UNTIL=unix] recovery_cell.sh
set -u
NAME=${NAME:?}; READERS=${READERS:?}; K=${K:?}; GAP_MS=${GAP_MS:?}; ARMS=${ARMS:?}
HOLD_GIB=${HOLD_GIB:?}; ARM_MB=${ARM_MB:?}; HOLD_S=${HOLD_S:-2}; ARM_WAIT_S=${ARM_WAIT_S:-45}
ROUNDS=${ROUNDS:-100000}; UNTIL=${UNTIL:-9999999999}; PAUSE_S=${PAUSE_S:-60}; SETTLE_S=${SETTLE_S:-20}
H=$(cd "$(dirname "$0")" && pwd)
R=$HOME/.cache/memra-1330
export TMPDIR=$R/tmp
mkdir -p "$TMPDIR"
OUT=$R/runs/$NAME
mkdir -p "$OUT"
STOP=$OUT/STOP
rm -f "$STOP" "$OUT/DONE"
echo $$ > "$OUT/cell.pid"
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
t0=$(date +%s)
log "cell $NAME READERS=$READERS K=$K GAP_MS=$GAP_MS ARMS=$ARMS HOLD_GIB=$HOLD_GIB HOLD_S=$HOLD_S ARM_MB=$ARM_MB ARM_WAIT_S=$ARM_WAIT_S PAUSE_S=$PAUSE_S SETTLE_S=$SETTLE_S ROUNDS=$ROUNDS UNTIL=$UNTIL"
log "host: $(hostname) $(uname -r) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) min_free_kbytes=$(cat /proc/sys/vm/min_free_kbytes) probe=$(sha256sum "$H/ctx_probe.py" | cut -c1-16)"
arms_csv=$(echo "$ARMS" | tr ' ' ',')
arms=($ARMS)
python3 "$H/memwatch.py" "$OUT/memwatch.jsonl" 50 "$STOP" > /dev/null 2>&1 < /dev/null &
echo $! > "$OUT/memwatch.pid"
stop_readers() {  # exactly the readers of this episode: each recorded subshell and its children
  touch "$ESTOP"
  for p in $(cat "$EPIDS" 2>/dev/null); do
    for c in $(ps -o pid= --ppid "$p" 2>/dev/null); do
      for g in $(ps -o pid= --ppid "$c" 2>/dev/null); do kill "$g" 2>/dev/null; done
      kill "$c" 2>/dev/null
    done
    kill "$p" 2>/dev/null
  done
}
go() { [ $r -lt "$ROUNDS" ] && [ "$(date +%s)" -lt "$UNTIL" ] && [ ! -e "$STOP" ]; }
r=0; ep=0; ESTOP=$OUT/stop.0; EPIDS=$OUT/reader.pids.0
trap 'stop_readers; touch "$STOP"; log "cell stopped by signal"; exit 1' TERM INT
while go; do
  ep=$((ep + 1))
  ESTOP=$OUT/stop.$ep; EPIDS=$OUT/reader.pids.$ep
  log "episode $ep starts"
  FILL_CAP_S=90 bash "$H/fill_live.sh" 3000 "$READERS" 100000 "$ESTOP" "$OUT/mem.log" "$EPIDS" >> "$OUT/run.log" 2>&1
  sleep "$SETTLE_S"
  while go; do
    r=$((r + 1))
    harm=${arms[$((r % ${#arms[@]}))]}
    if [ "$HOLD_GIB" -gt 0 ]; then
      line=$(python3 "$H/ctx_probe.py" hold "$HOLD_GIB" "$HOLD_S" "$harm" 2>> "$OUT/hold.log" < /dev/null | tail -n 1)
      echo "{\"round\": $r, \"episode\": $ep, \"rec\": ${line:-null}}" >> "$OUT/holders.jsonl"
      case $line in *'"released"'*) ;; *) log "round $r: holder failed: $(tail -n 1 "$OUT/hold.log")" ;; esac
    fi
    python3 "$H/ctx_probe.py" round "$OUT/creations.jsonl" "$r" "$K" "$GAP_MS" "$ARM_MB" "$ARM_WAIT_S" "$arms_csv" < /dev/null 2>&1 | tee -a "$OUT/run.log"
    if [ "${PIPESTATUS[0]}" = 3 ]; then log "episode $ep ends at round $r: MemFree stayed over $ARM_MB MB"; break; fi
  done
  stop_readers
  go && sleep "$PAUSE_S"
done
log "end: rounds=$r episodes=$ep $(grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';')"
touch "$STOP"
journalctl -k --since "@$t0" --no-pager -o short-unix 2>/dev/null | grep NVRM > "$OUT/nvrm.log"
log "creations=$(grep -vc skipped "$OUT/creations.jsonl") first_try_failed=$(grep -c '"failed_first": true' "$OUT/creations.jsonl") nvrm_lines=$(wc -l < "$OUT/nvrm.log") kgrctx=$(grep -c kgrctxAllocCtxBuffers "$OUT/nvrm.log")"
touch "$OUT/DONE"
