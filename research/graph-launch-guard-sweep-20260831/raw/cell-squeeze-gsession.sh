#!/bin/bash
# CELL: guard-fires squeeze on the GRAPH-SESSION route (lane/graph-launch-guard-sweep-20260831).
#
# ROUTE STATEMENT: decode.rs GraphSession::step (site decode.rs:75), the worker's
# solo-interactive whole-step replay - the ONE captured-graph route a box12-shaped
# q38+DFlash2 deployment actually reaches in serving (dspark disables MTP spec; the
# DFlash2 drafter never arms the verify-graph pool). This route has NO eager twin by
# construction (the session IS the graph), so the guard's contract is a RECOVERABLE
# session-scoped refusal: `[graph-session] graph replay suspended:` + an error event
# on THIS session; the process and every peer session live.
#
# Boot shape: plain q38 (no spec vars) so a solo greedy interactive request rides
# GraphSession replay. g1 LESSON: GraphSession degrades to batched-eager the moment a
# second session admits, so the birth-race storm REMOVES the route it is trying to
# squeeze; and the external-ballast "wall" (~650MB nvidia-smi free, where even 16MB
# foreign chunks fail) IS driver exhaustion - mem_get_info free is already below the
# floor there while nvidia-smi still reads hundreds of MB. So this cell keeps the box
# SOLO: one long greedy generation stepping its captured graph, one chase ballast
# walking to the wall, nothing else. At the wall the session's next step() refuses
# recoverably.
# Usage: cell-squeeze-gsession.sh <run-tag>
set -u
. /home/ubuntu/guard-lane/gl-lib.sh
SERVE_ENV="$SERVE_ENV_COMMON"
TAG=${1:?run tag}
LOG=serve-gsess-$TAG.log
resp_ok() { python3 -c "import json,sys;d=json.load(open('$1'));t=d.get('text') or (d.get('choices') or [{}])[0].get('text');print('OK:'+str(len(t)) if t else 'FAIL:'+str(d)[:120])" 2>/dev/null || echo PARSE_FAIL; }

say "=== GRAPH-SESSION SQUEEZE RUN $TAG (bin=lane) ==="
gpu_empty || { say "REFUSING: GPU not empty / server alive"; exit 2; }
dmesg_mark
sampler_start $G/free-gsess-$TAG.csv
nohup $BALLAST 0 64000 > $G/ballast-pre-$TAG.log 2>&1 &
BPRE_PID=$!
sleep 8
# MEMRA_SERVE_SPEC=0 (g1 lesson, second half): the q38 artifact CARRIES an MTP block,
# so with spec serving on a solo greedy request rides SpecSession, never GraphSession;
# the plain-decode box12 exposure this cell exists for needs spec OFF.
boot "$BINLANE" "MEMRA_ADMIT_RESERVE_MB=16 MEMRA_SERVE_SPEC=0" "$LOG" || { sampler_stop; kill -9 $BPRE_PID 2>/dev/null; exit 1; }

# ---- phase 1: solo greedy request rides GraphSession; false-positive check ----
req 0 128 0 "" $G/resp-$TAG-p1a.json
S=$(suspended_count $LOG)
[ "${S:-0}" != "0" ] && { say "FALSE-POSITIVE at healthy headroom"; shutdown_srv; sampler_stop; exit 9; }
say "phase1 clean: suspended=0 free=$(gpu0_free_mb)MB (graph-session route armed for solo greedy interactive)"

# ---- phase 2: ONE long solo greedy generation (graph session) + ballast to the wall ----
rm -f $G/.stop-$TAG
( n=0
  while [ ! -f $G/.stop-$TAG ]; do
    n=$((n+1))
    req "u$n" 4900 0 "" $G/resp-$TAG-loop$n.json
    echo $n > $G/.loops-$TAG
    sleep 1
  done ) &
LOOP_PID=$!
sleep 6

say "chase ballast to the wall (driver exhaustion): free=$(gpu0_free_mb)MB (hold to free-2000, then 64MB/1s retrying chase)"
FREE=$(gpu0_free_mb)
HOLD=$((FREE - 2000)); [ $HOLD -lt 256 ] && HOLD=256
nohup $BALLAST 0 $HOLD 64 1 $((FREE + 50000)) > $G/ballast-$TAG.log 2>&1 &
BPID=$!

FIRED_AT=""
for i in $(seq 1 210); do
  S=$(grep -ac "\[graph-session\] graph replay suspended:" $G/$LOG || true)
  if [ "${S:-0}" != "0" ]; then
    FIRED_AT="t=$((i*2))s"
    say "graph-session suspension observed ($FIRED_AT)"
    break
  fi
  sleep 2
done

sleep 10
touch $G/.stop-$TAG
ALIVE=no; kill -0 $SRV_PID 2>/dev/null && ALIVE=yes
SESSERR=$(grep -ac "graph-session replay refused" $G/$LOG || true)
say "server alive: $ALIVE ; refusal errors: $SESSERR"

# ---- phase 3: recovery ----
kill -TERM $BPID 2>/dev/null
kill -TERM ${BPRE_PID:-0} 2>/dev/null
wait $LOOP_PID 2>/dev/null
sleep 5
req 7 96 0 "" $G/resp-$TAG-recovery.json
RECOV=$(resp_ok $G/resp-$TAG-recovery.json)

SUS=$(grep -ac "\[graph-session\] graph replay suspended:" $G/$LOG || true)
shutdown_srv
sampler_stop
dmesg_check gsess-$TAG
FAULTS=$(grep -c . $G/dmesg-gsess-$TAG.txt 2>/dev/null | head -1); FAULTS=${FAULTS:-0}
say "VERDICT $TAG: gsession_suspended=$SUS refusals=$SESSERR fired_at=${FIRED_AT:-NEVER} recovery=$RECOV alive=$ALIVE dmesg_faults=$FAULTS"
rm -f $G/.stop-$TAG $G/.loops-$TAG
if [ "${SUS:-0}" != "0" ] && [ "$FAULTS" = "0" ] && [ "$ALIVE" = "yes" ] && [ "${RECOV%%:*}" = "OK" ]; then
  say "RUN $TAG: PASS"; exit 0
fi
say "RUN $TAG: FAIL"; exit 1
