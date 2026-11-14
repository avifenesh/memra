#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): one cell of driver-level boots
# (ctx_probe.py), written to ~/.cache/memra-1316/runs/NAME/.
#   STATE=asis   boots in the host's state as found (the control)
#   STATE=live   page cache filled to MemFree < 3000 MB, READERS readers kept running (fill_live.sh)
#   MODE=loop    N boots, GAP_MS apart
#   MODE=exit    TRIALS times: a holder process backs HOLD_GIB of device memory for HOLD_S seconds
#                and exits cleanly (a server that stops), then one boot every second for BURST
#                seconds; each boot's record carries the seconds since that exit
#   MODE=dip     TRIALS times: the same holder exits, then for BURST seconds armed boots one after
#                another: each waits until MemFree is under ARM_MB (the readers have used up what
#                the holder freed) and creates its context at that moment
#   MODE=churn   one process creates and destroys a context over and over for SECONDS_RUN (about
#                five a second), each a sample of the allocation a boot needs; HOLD_GIB > 0 adds a
#                holder that backs HOLD_GIB, keeps it HOLD_S, exits and starts again HOLD_GAP_S later
# RETRY_MAX > 0 repeats a call that returned CUDA_ERROR_OUT_OF_MEMORY in the same process, 50 ms
# apart (the first try is still the boot's verdict).
# The kernel's NVRM lines for the cell's window go to nvrm.log. Reads only; changes no setting.
# usage: NAME=x STATE=asis|live MODE=loop|exit|dip [N=300 GAP_MS=200] [TRIALS=20 HOLD_GIB=30 HOLD_S=10 BURST=20] [ARM_MB=800]
#        [READERS=2] [RETRY_MAX=0] probe_cell.sh
set -u
NAME=${NAME:?}; STATE=${STATE:?}; MODE=${MODE:?}
N=${N:-300}; GAP_MS=${GAP_MS:-200}; TRIALS=${TRIALS:-20}; HOLD_GIB=${HOLD_GIB:-30}; HOLD_S=${HOLD_S:-10}
BURST=${BURST:-20}; READERS=${READERS:-2}; RETRY_MAX=${RETRY_MAX:-0}; ARM_MB=${ARM_MB:-800}; SECONDS_RUN=${SECONDS_RUN:-240}; HOLD_GAP_S=${HOLD_GAP_S:-20}
H=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/.cache/memra-1316/runs/$NAME
mkdir -p "$OUT"
STOP=$OUT/STOP; PIDS=$OUT/reader.pids
rm -f "$STOP" "$PIDS" "$OUT/DONE"
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
t0=$(date +%s)
log "cell $NAME STATE=$STATE MODE=$MODE N=$N GAP_MS=$GAP_MS TRIALS=$TRIALS HOLD_GIB=$HOLD_GIB HOLD_S=$HOLD_S BURST=$BURST READERS=$READERS RETRY_MAX=$RETRY_MAX"
log "host: $(uname -r) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) min_free_kbytes=$(cat /proc/sys/vm/min_free_kbytes)"
if [ "$STATE" = live ]; then
  bash "$H/fill_live.sh" 3000 "$READERS" 3000 "$STOP" "$OUT/mem.log" "$PIDS" 2>&1 | tee -a "$OUT/run.log"
fi
log "start: $(grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';')"
case $MODE in
  loop)
    python3 "$H/ctx_probe.py" loop "$OUT/boots.jsonl" "$N" "$GAP_MS" "$RETRY_MAX" 2>&1 | tee -a "$OUT/run.log" ;;
  exit)
    for tr in $(seq 1 "$TRIALS"); do
      python3 "$H/ctx_probe.py" hold "$HOLD_GIB" "$HOLD_S" >> "$OUT/hold.log" 2>&1
      hrc=$?
      rel=$(date +%s.%N)
      [ $hrc = 0 ] || log "trial $tr: holder rc=$hrc $(tail -n 1 "$OUT/hold.log")"
      for b in $(seq 1 "$BURST"); do
        now=$(date +%s.%N)
        line=$(timeout 120 python3 "$H/ctx_probe.py" one "$RETRY_MAX" 2>/dev/null | tail -n 1)
        [ -n "$line" ] || line='{"ok": false, "first_fail": "probe died"}'
        echo "{\"trial\": $tr, \"since_exit_s\": $(echo "$now - $rel" | bc | sed 's/^\./0./'), \"rec\": $line}" >> "$OUT/boots.jsonl"
        case $line in *'"ok": false'*) log "trial $tr boot $b: FAIL" ;; esac
        sleep 1
      done
    done ;;
  dip)
    for tr in $(seq 1 "$TRIALS"); do
      python3 "$H/ctx_probe.py" hold "$HOLD_GIB" "$HOLD_S" >> "$OUT/hold.log" 2>&1
      hrc=$?
      rel=$(date +%s.%N)
      [ $hrc = 0 ] || log "trial $tr: holder rc=$hrc $(tail -n 1 "$OUT/hold.log")"
      end=$((SECONDS + BURST))
      while [ $SECONDS -lt $end ]; do
        line=$(timeout 120 python3 "$H/ctx_probe.py" armed "$ARM_MB" $((end - SECONDS + 1)) "$RETRY_MAX" 2>/dev/null | tail -n 1)
        [ -n "$line" ] || line='{"ok": false, "first_fail": "probe died"}'
        echo "{\"trial\": $tr, \"since_exit_s\": $(echo "$(date +%s.%N) - $rel" | bc | sed 's/^\./0./'), \"rec\": $line}" >> "$OUT/boots.jsonl"
        case $line in *'"ok": false'*) log "trial $tr: FAIL" ;; esac
      done
    done ;;
  churn)
    # One process creating and destroying contexts for SECONDS. With HOLD_GIB > 0 a holder
    # process backs that much device memory, keeps it HOLD_S seconds and exits, again and again,
    # for the whole cell (a server that stops, over and over); its exits go to hold.log.
    if [ "$HOLD_GIB" -gt 0 ]; then
      (
        while [ ! -e "$STOP" ]; do
          python3 "$H/ctx_probe.py" hold "$HOLD_GIB" "$HOLD_S" >> "$OUT/hold.log" 2>&1
          sleep "$HOLD_GAP_S"
        done
      ) < /dev/null &
      echo $! >> "$PIDS"
    fi
    python3 "$H/ctx_probe.py" churn "$OUT/boots.jsonl" "$SECONDS_RUN" "$RETRY_MAX" 2>&1 | tee -a "$OUT/run.log" ;;
  *) log "MODE $MODE unknown"; exit 2 ;;
esac
log "end: $(grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';')"
touch "$STOP"
if [ -s "$PIDS" ]; then
  # Stop exactly the readers this cell started: each recorded subshell and its children.
  for p in $(cat "$PIDS"); do
    for c in $(ps -o pid= --ppid "$p" 2>/dev/null); do
      for g in $(ps -o pid= --ppid "$c" 2>/dev/null); do kill "$g" 2>/dev/null; done
      kill "$c" 2>/dev/null
    done
    kill "$p" 2>/dev/null
  done
fi
journalctl -k --since "@$t0" --no-pager -o short-unix 2>/dev/null | grep NVRM > "$OUT/nvrm.log"
log "boots=$(wc -l < "$OUT/boots.jsonl") failed=$(grep -c -E '"ok": false|"rc": 2' "$OUT/boots.jsonl") nvrm_lines=$(wc -l < "$OUT/nvrm.log")"
touch "$OUT/DONE"
