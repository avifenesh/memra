#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): one cell of restarts at the moment
# both occurrences booted: a process that held device memory has exited and the readers have just
# used up what it freed. Written to ~/.cache/memra-1316/runs/NAME/.
# The page cache is filled to MemFree < 3000 MB and READERS readers kept running (fill_live.sh).
# Then rounds, until UNTIL (a unix time) or ROUNDS rounds:
#   a holder backs HOLD_GIB of device memory, keeps it HOLD_S seconds and exits cleanly
#   (ctx_probe.py hold: a server that stops);
#   the round waits (45 s at most) until MemFree is back under ARM_MB;
#   K boot-only starts of the real server follow, GAP_MS apart, the arms of ARMS in turn
#   (server_boot.py); the first arm alternates between rounds.
# When MemFree does not come back under ARM_MB the round makes no boot and the episode ends: the
# readers stop for PAUSE_S seconds (60) and start again. EP_ROUNDS=n ends every episode after n
# rounds; with HOLD_GIB=0 there is no holder, so EP_ROUNDS=1 HOLD_GIB=0 gives the plainest
# restart: readers off for PAUSE_S, readers on until MemFree is at the floor and then SETTLE_S
# seconds more (0 unless set), K boots.
# boots.jsonl has one row per boot with its round; memwatch.jsonl MemFree, MemAvailable, Cached
# and the large free blocks every 50 ms; nvrm.log the kernel's NVRM lines. Reads only; changes
# no setting.
# usage: NAME=x READERS=n K=n GAP_MS=n ARMS="B T" HOLD_GIB=n ARM_MB=n [HOLD_S=2] [PAUSE_S=60] [SETTLE_S=0] [EP_ROUNDS=n] [ROUNDS=n] [UNTIL=unix] transition_cell.sh
set -u
NAME=${NAME:?}; READERS=${READERS:?}; K=${K:?}; GAP_MS=${GAP_MS:?}; ARMS=${ARMS:?}
HOLD_GIB=${HOLD_GIB:?}; ARM_MB=${ARM_MB:?}; HOLD_S=${HOLD_S:-2}; ROUNDS=${ROUNDS:-100000}; UNTIL=${UNTIL:-9999999999}
PAUSE_S=${PAUSE_S:-60}; EP_ROUNDS=${EP_ROUNDS:-100000}; SETTLE_S=${SETTLE_S:-0}
H=$(cd "$(dirname "$0")" && pwd)
R=$HOME/.cache/memra-1316
OUT=$R/runs/$NAME
mkdir -p "$OUT"
STOP=$OUT/STOP
rm -f "$STOP" "$OUT/DONE"
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
t0=$(date +%s)
log "cell $NAME READERS=$READERS K=$K GAP_MS=$GAP_MS ARMS=$ARMS HOLD_GIB=$HOLD_GIB HOLD_S=$HOLD_S ARM_MB=$ARM_MB PAUSE_S=$PAUSE_S SETTLE_S=$SETTLE_S EP_ROUNDS=$EP_ROUNDS ROUNDS=$ROUNDS UNTIL=$UNTIL"
log "host: $(uname -r) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) min_free_kbytes=$(cat /proc/sys/vm/min_free_kbytes)"
arms=($ARMS)
for a in $ARMS; do
  [ -x "$R/bins/$a/memra-server" ] || { log "CELL-FAIL no binary for arm $a"; exit 1; }
  log "arm $a: $(sha256sum "$R/bins/$a/memra-server" | cut -c1-64) runtime sources: $(tr '\n' ' ' < "$R/bins/$a/runtime-src.sha" | sed 's|crates/memra-runtime/src/||g')"
done
python3 "$H/memwatch.py" "$OUT/memwatch.jsonl" 50 "$STOP" > /dev/null 2>&1 < /dev/null &
. "$H/serve_env.sh"
env | grep '^MEMRA_' | sed "s|$HOME|~|g" | sort > "$OUT/env.txt"
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
r=0; ep=0
while [ $r -lt "$ROUNDS" ] && [ "$(date +%s)" -lt "$UNTIL" ]; do
  # An episode: readers on (the fill gives up after 90 s), then rounds until MemFree no longer
  # comes back under ARM_MB after an exit. After a dozen or two boots at the floor the kernel
  # keeps several GB free for as long as the readers run (campaign G), so the episode then ends:
  # readers off for PAUSE_S seconds, and a new episode starts.
  ep=$((ep + 1))
  ESTOP=$OUT/stop.$ep; EPIDS=$OUT/reader.pids.$ep
  er=0
  # Appended, not piped: the fill's last `cat`s outlive it until their file ends, and a pipe
  # would hold this script until they do (40 to 70 s here), long after the floor was reached.
  FILL_CAP_S=90 bash "$H/fill_live.sh" 3000 "$READERS" 3000 "$ESTOP" "$OUT/mem.log" "$EPIDS" >> "$OUT/run.log" 2>&1
  # The readers then run SETTLE_S seconds with no context being created: every failure in the
  # cells before came at the first creation after 40 s or more of that.
  sleep "$SETTLE_S"
  while [ $r -lt "$ROUNDS" ] && [ "$(date +%s)" -lt "$UNTIL" ]; do
    r=$((r + 1))
    # Rotate the arms so each one gets the first boot of a round equally often.
    bins=""
    for j in $(seq 0 $((${#arms[@]} - 1))); do
      a=${arms[$(((j + r - 1) % ${#arms[@]}))]}
      bins="$bins${bins:+,}$R/bins/$a"
    done
    if [ "$HOLD_GIB" -gt 0 ]; then
      python3 "$H/ctx_probe.py" hold "$HOLD_GIB" "$HOLD_S" >> "$OUT/hold.log" 2>&1 || log "round $r: holder failed: $(tail -n 1 "$OUT/hold.log")"
    fi
    ROUND=$r ARM_MB=$ARM_MB ARM_WAIT_S=45 ARM_SKIP=1 python3 "$H/server_boot.py" "$OUT/boots.jsonl" "$K" "$GAP_MS" "$bins" 2>&1 | tee -a "$OUT/run.log"
    if [ "${PIPESTATUS[0]}" = 3 ]; then log "episode $ep ends at round $r: MemFree stayed over $ARM_MB MB"; break; fi
    er=$((er + 1))
    [ $er -ge "$EP_ROUNDS" ] && break
  done
  stop_readers
  [ "$(date +%s)" -lt "$UNTIL" ] && sleep "$PAUSE_S"
done
log "end: rounds=$r episodes=$ep $(grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';')"
touch "$STOP"
journalctl -k --since "@$t0" --no-pager -o short-unix 2>/dev/null | grep NVRM > "$OUT/nvrm.log"
log "boots=$(wc -l < "$OUT/boots.jsonl") died=$(grep -c '"outcome": "fatal"' "$OUT/boots.jsonl") first_try_failed=$(grep -c '"first_try_failed": true' "$OUT/boots.jsonl") nvrm_lines=$(wc -l < "$OUT/nvrm.log") kgrctx=$(grep -c kgrctxAllocCtxBuffers "$OUT/nvrm.log")"
touch "$OUT/DONE"
