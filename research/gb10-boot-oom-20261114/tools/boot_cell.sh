#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): one cell of boot-only starts of the
# real memra-server (server_boot.py), written to ~/.cache/memra-1316/runs/NAME/.
#   READERS=0    the host as found (the control)
#   READERS>0    page cache filled to MemFree < 3000 MB and READERS readers kept running
#                (fill_live.sh) for the whole cell
#   HOLD_GIB>0   also a holder process that backs HOLD_GIB of device memory, keeps it HOLD_S
#                seconds, exits and starts again HOLD_GAP_S later (ctx_probe.py hold), all cell long
# N boots, GAP_MS apart, the binaries of ARMS taken in turn (bins/ARM/memra-server): each boot is
# the serving of serve_env.sh up to the worker's `Engine ready` line, where the process is killed,
# or its own exit on a FATAL. MemFree, MemAvailable, Cached and the large free blocks go to
# memwatch.jsonl every 50 ms, the kernel's NVRM lines for the cell's window to nvrm.log.
# Reads only; changes no setting.
# usage: NAME=x READERS=n N=n GAP_MS=n ARMS="B T" [HOLD_GIB=n HOLD_S=3 HOLD_GAP_S=25] boot_cell.sh
set -u
NAME=${NAME:?}; READERS=${READERS:?}; N=${N:?}; GAP_MS=${GAP_MS:?}; ARMS=${ARMS:?}
HOLD_GIB=${HOLD_GIB:-0}; HOLD_S=${HOLD_S:-3}; HOLD_GAP_S=${HOLD_GAP_S:-25}
H=$(cd "$(dirname "$0")" && pwd)
R=$HOME/.cache/memra-1316
OUT=$R/runs/$NAME
mkdir -p "$OUT"
STOP=$OUT/STOP; PIDS=$OUT/reader.pids
rm -f "$STOP" "$PIDS" "$OUT/DONE"
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
t0=$(date +%s)
log "cell $NAME READERS=$READERS N=$N GAP_MS=$GAP_MS ARMS=$ARMS HOLD_GIB=$HOLD_GIB HOLD_S=$HOLD_S HOLD_GAP_S=$HOLD_GAP_S"
log "host: $(uname -r) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) min_free_kbytes=$(cat /proc/sys/vm/min_free_kbytes)"
bins=""
for a in $ARMS; do
  [ -x "$R/bins/$a/memra-server" ] || { log "CELL-FAIL no binary for arm $a"; exit 1; }
  log "arm $a: $(sha256sum "$R/bins/$a/memra-server" | cut -c1-64) runtime sources: $(tr '\n' ' ' < "$R/bins/$a/runtime-src.sha" | sed 's|crates/memra-runtime/src/||g')"
  bins="$bins${bins:+,}$R/bins/$a"
done
if [ "$READERS" -gt 0 ]; then
  bash "$H/fill_live.sh" 3000 "$READERS" 3000 "$STOP" "$OUT/mem.log" "$PIDS" 2>&1 | tee -a "$OUT/run.log"
fi
if [ "$HOLD_GIB" -gt 0 ]; then
  # A process that backs HOLD_GIB of device memory, keeps it HOLD_S seconds, exits cleanly and
  # starts again HOLD_GAP_S later, for the whole cell: a server that stops, over and over, so the
  # readers refill what it freed again and again (the state both occurrences booted in).
  (
    while [ ! -e "$STOP" ]; do
      python3 "$H/ctx_probe.py" hold "$HOLD_GIB" "$HOLD_S" >> "$OUT/hold.log" 2>&1
      [ -e "$STOP" ] || sleep "$HOLD_GAP_S"
    done
  ) < /dev/null &
  echo $! >> "$PIDS"
fi
python3 "$H/memwatch.py" "$OUT/memwatch.jsonl" 50 "$STOP" > /dev/null 2>&1 < /dev/null &
log "start: $(grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | tr -s ' ' | tr '\n' ';')"
(
  . "$H/serve_env.sh"
  env | grep '^MEMRA_' | sed "s|$HOME|~|g" | sort > "$OUT/env.txt"
  python3 "$H/server_boot.py" "$OUT/boots.jsonl" "$N" "$GAP_MS" "$bins"
) 2>&1 | tee -a "$OUT/run.log"
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
log "boots=$(wc -l < "$OUT/boots.jsonl") died=$(grep -c '"outcome": "fatal"' "$OUT/boots.jsonl") first_try_failed=$(grep -c '"first_try_failed": true' "$OUT/boots.jsonl") nvrm_lines=$(wc -l < "$OUT/nvrm.log") kgrctx=$(grep -c kgrctxAllocCtxBuffers "$OUT/nvrm.log")"
touch "$OUT/DONE"
