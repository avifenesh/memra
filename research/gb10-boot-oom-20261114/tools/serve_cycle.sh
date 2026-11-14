#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease (memra#1316): a chain of serving cycles, the
# sequence both occurrences came from (memra#1307's DFlash2 cells with live readers, one after
# another). Each cycle:
#   boot memra-server of the cycle's arm (serve1316.sh) and wait for /readyz or for its exit;
#   when ready: fill the page cache to MemFree < 3000 MB and keep two readers going
#   (fill_live.sh), send the cell's requests (client1316.py), tell the readers to stop (they stop
#   at their next file boundary, at most 30 s later, so some are still reading at the next boot),
#   SIGTERM the server and wait for it to go;
#   start the next cycle GAP_S seconds (15) after the SIGTERM, as memra#1307's cells did.
# ARMS is cycled in order (bins/ARM/memra-server). cycles.jsonl gets one row per boot: the arm,
# ready or fatal, the seconds to ready, the server's FATAL and [ctx-retry] lines. memwatch.jsonl
# has MemFree, MemAvailable, Cached and the large free blocks every 50 ms for the whole cell;
# nvrm.log the kernel's NVRM lines. Reads only; changes no setting.
# usage: NAME=x CYCLES=n ARMS="T B" [GAP_S=15] [REQS=1] serve_cycle.sh
set -u
NAME=${NAME:?}; CYCLES=${CYCLES:?}; ARMS=${ARMS:?}; GAP_S=${GAP_S:-15}; REQS=${REQS:-1}
H=$(cd "$(dirname "$0")" && pwd)
R=$HOME/.cache/memra-1316
CD=$R/runs/$NAME; mkdir -p "$CD"
log() { echo "$(date -Is) $*" | tee -a "$CD/run.log"; }
mi() { grep -E '^(MemFree|MemAvailable|Cached):' /proc/meminfo | awk '{printf "%s %d ", $1, $2/1024}'; }
t0=$(date +%s)
WSTOP=$CD/memwatch.stop; rm -f "$WSTOP"
python3 "$H/memwatch.py" "$CD/memwatch.jsonl" 50 "$WSTOP" > /dev/null 2>&1 < /dev/null &
arms=($ARMS)
log "cell $NAME CYCLES=$CYCLES ARMS=$ARMS GAP_S=$GAP_S REQS=$REQS host: $(uname -r) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) $(mi)"
for i in $(seq 1 "$CYCLES"); do
  arm=${arms[$(((i - 1) % ${#arms[@]}))]}
  c=$(printf '%02d' "$i")-$arm
  OUT=$CD/$c; mkdir -p "$OUT"
  STOP=$OUT/stop; PIDS=$OUT/reader.pids
  tb=$(date +%s.%N)
  log "$c boot: $(mi)"
  OUT=$OUT BIN=$R/bins/$arm bash "$H/serve1316.sh" &
  sp=$!
  waitfor --file "$OUT/server.pid" --timeout 1m --every 1s --quiet > /dev/null 2>&1
  spid=$(cat "$OUT/server.pid")
  waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' --fail-if "! kill -0 $spid" \
    --every 1s --timeout 10m --quiet >> "$OUT/run.log" 2>&1
  wrc=$?
  tr=$(date +%s.%N)
  if [ $wrc = 0 ]; then
    outcome=ready
    if [ "$REQS" = 1 ]; then
      bash "$H/fill_live.sh" 3000 2 900 "$STOP" "$OUT/memfree.log" "$PIDS" >> "$OUT/run.log" 2>&1
      python3 "$H/client1316.py" "$OUT/rows.jsonl" "$c" >> "$OUT/run.log" 2>&1
    fi
    touch "$STOP"
    kill -TERM "$spid"
    tt=$(date +%s.%N)
    waitfor --gone "$spid" --timeout 5m --every 1s --quiet >> "$OUT/run.log" 2>&1 || kill -KILL "$spid"
  else
    outcome=fatal
    touch "$STOP"
    kill -TERM "$spid" 2>/dev/null
    tt=$(date +%s.%N)
  fi
  wait $sp
  python3 - "$CD/cycles.jsonl" "$c" "$arm" "$outcome" "$tb" "$tr" "$OUT/server.log" <<'PY'
import json, sys
out, c, arm, outcome, tb, tr, logp = sys.argv[1:8]
lines = open(logp, errors="replace").read().splitlines()
row = {"cycle": c, "arm": arm, "outcome": outcome, "t_boot": float(tb), "wait_s": round(float(tr) - float(tb), 2),
       "fatal": [l for l in lines if "FATAL" in l][:3],
       "ctx_retry": [l for l in lines if "[ctx-retry]" in l],
       "engine_ready": [l.split()[0] for l in lines if "Engine ready" in l][:1]}
open(out, "a").write(json.dumps(row) + "\n")
print(json.dumps(row)[:600])
PY
  log "$c $outcome: $(mi)"
  # The next boot starts GAP_S after this cycle's SIGTERM.
  python3 -c "import sys, time; time.sleep(max(0.0, float(sys.argv[1]) + float(sys.argv[2]) - time.time()))" "$tt" "$GAP_S"
done
touch "$WSTOP"
# Readers stop by themselves at their next file boundary (30 s at most); wait for the last ones.
python3 -c "import time; time.sleep(31)"
journalctl -k --since "@$t0" --no-pager -o short-unix 2>/dev/null | grep NVRM > "$CD/nvrm.log"
log "done: cycles=$(wc -l < "$CD/cycles.jsonl") fatal=$(grep -c '"outcome": "fatal"' "$CD/cycles.jsonl") with_retry=$(grep -c 'ctx-retry' "$CD/cycles.jsonl") nvrm_lines=$(wc -l < "$CD/nvrm.log")"
