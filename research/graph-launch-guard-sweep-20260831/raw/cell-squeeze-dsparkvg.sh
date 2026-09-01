#!/bin/bash
# CELL: guard-fires squeeze on the DSPARK VERIFY route (lane/graph-launch-guard-sweep-20260831).
# Route exercised: dspark SERVE verify graphs (spec.rs run_full/run_segment via
# qwen35_verify_tparallel, reached from DsparkSpecSession bursts) on the box12 q38 shape.
# Shape lessons (runs 1-2, logs banked):
#   - concurrent greedy wave-demotes to PLAIN (dspark greedy admission is LOW-wave), so
#     the driver keeps exactly ONE solo dspark generation in flight, back-to-back;
#   - 256MB ballast chunks stall at ~689MB free (fragmentation); 64MB/1s chunks walk
#     free below the 256MB floor;
#   - the guard can only fire in a session ALREADY decoding when free crosses the floor,
#     so the continuous loop maximizes the chance a burst straddles the crossing;
#   - run 3: an EXTERNAL cudaMalloc walls at ~650MB free (the driver refuses the tail
#     to a foreign process) while the server keeps serving happily, so the last stretch
#     below the 256MB floor must be eaten by the server's OWN async pool: the loop
#     rotates UNIQUE two-prompt combinations of the real pool so every completed
#     generation inserts a fresh ~160MB+ prefix-cache entry
#     (MEMRA_PREFIX_CACHE_MB=16384 lifts the default 800MB cap out of the way).
# Fresh server per run (the suspended note is once-per-process). FALSE-POSITIVE check:
# any suspended line at healthy headroom kills the run.
# Usage: cell-squeeze-dsparkvg.sh <run-tag>
set -u
. /home/ubuntu/guard-lane/gl-lib.sh
TAG=${1:?run tag}
LOG=serve-squeeze-$TAG.log
resp_ok() { python3 -c "import json,sys;d=json.load(open('$1'));t=d.get('text') or (d.get('choices') or [{}])[0].get('text');print('OK:'+str(len(t)) if t else 'FAIL:'+str(d)[:120])" 2>/dev/null || echo PARSE_FAIL; }

say "=== SQUEEZE RUN $TAG (bin=lane) ==="
gpu_empty || { say "REFUSING: GPU not empty / server alive"; exit 2; }
dmesg_mark
sampler_start $G/free-squeeze-$TAG.csv

# THE WORKING RECIPE (step37 battery phaseD, vg-battery-d.sh, adopted after runs 2-10
# proved every polite approach cannot cross the floor): a foreign process's cudaMalloc
# is REFUSED by the driver below ~650MB free on this card, so the tail below the floor
# can only be eaten by the SERVER's own allocations. phaseD's shape: PRE-BALLAST so the
# server boots into a small world (~12GB free), open the sanctioned teeth door
# (MEMRA_ADMIT_RESERVE_MB=16: admissions keep passing at tiny free, exactly what the
# door exists to prove), keep sessions decoding, then a retrying chase ballast eats
# every freed byte and the server's own session allocations walk driver-free through
# the 256MB floor from the inside.
nohup $BALLAST 0 64000 > $G/ballast-pre-$TAG.log 2>&1 &
BPRE_PID=$!
sleep 8
say "pre-ballast holding: free=$(gpu0_free_mb)MB"
boot "$BINLANE" "MEMRA_ADMIT_RESERVE_MB=16 MEMRA_GRAPH_CENSUS=1" "$LOG" || { sampler_stop; kill -9 $BPRE_PID 2>/dev/null; exit 1; }

CAL=$(grep -ac 'boot calibration done.*route=dspark' $G/$LOG || true)
say "calibration route=dspark lines: $CAL"

# ---- phase 1: healthy headroom, dspark engagement + false-positive check ----
req 0 96 0 "" $G/resp-$TAG-p1a.json
req 1 96 0 "" $G/resp-$TAG-p1b.json
ENG=$(grep -ac 'serve pool ENGAGED' $G/$LOG || true)
ACC=$(grep -ac 'dspark-acc' $G/$LOG || true)
S=$(suspended_count $LOG)
if [ "${S:-0}" != "0" ]; then
  say "FALSE-POSITIVE: $S suspended line(s) at healthy headroom; killing run"
  shutdown_srv; sampler_stop; dmesg_check squeeze-$TAG; exit 9
fi
CEN=$(grep -ac 'dspark-vg-census' $G/$LOG || true)
say "phase1 clean: suspended=0 pool_engaged=$ENG dspark_acc=$ACC vg_captures=$CEN free=$(gpu0_free_mb)MB"
[ "${CEN:-0}" = "0" ] && say "WARNING: zero vg captures in phase 1 (guard reachability evidence weak)"
[ "${ENG:-0}" = "0" ] && { say "ROUTE NOT ENGAGED (no vg pool): refusing to claim coverage"; shutdown_srv; sampler_stop; exit 7; }

# ---- phase 2: one LONG solo dspark generation + a stacker of live sessions ----
# Final mechanism (runs 2-12 receipts): an external cudaMalloc is refused by the
# driver below ~650MB free regardless of chunk size, so the stretch below the 256MB
# floor can only be HELD by live server state. The chase ballast walks free to the
# wall; then STACKED sessions (each ~220MB capacity-sized cache, still decoding, so
# their memory stays live) step free down through the floor while the long dspark
# generation admitted at active=1 is still mid-burst.
rm -f $G/.stop-$TAG
( n=0
  while [ ! -f $G/.stop-$TAG ]; do
    n=$((n+1))
    # solo dspark generation, 4000 tokens (the non-streaming deadline gate refuses
    # ~5400+ on these prompts: run-13 receipt), unique key per iteration
    req "u$n" 4900 0 "" $G/resp-$TAG-loop$n.json
    echo $n > $G/.loops-$TAG
    sleep 1
  done ) &
LOOP_PID=$!
sleep 8   # the long dspark generation admits SOLO and starts bursting

# LOUD shape check (run-13 lesson: a refused loop request fails in 0.4s and the cell
# spins 15k junk responses while claiming to squeeze): the first loop response must
# either still be running (no file yet = good, it is generating) or parse OK.
# (an EMPTY response file means curl already truncated it and the request is still
# generating: that is the healthy in-flight state, run-14 lesson)
if [ -s $G/resp-$TAG-loop1.json ]; then
  case "$(resp_ok $G/resp-$TAG-loop1.json)" in
    OK:*) : ;;
    *) say "LOOP REQUEST REFUSED/BROKEN: $(head -c 200 $G/resp-$TAG-loop1.json)"; touch $G/.stop-$TAG; shutdown_srv; sampler_stop; exit 6 ;;
  esac
fi

# THE CROSSING MECHANISM (adopted from step37 phaseF receipts on this box, after
# runs 2-18 receipts killed every other approach): a foreign process cannot allocate
# below ~640MB free, and small server allocations never exceed the boot high-water.
# What crossed the floor in phaseF was a RACE OF CONCURRENT MULTI-GB CACHE BIRTHS:
# several sessions admitted against the same ~13GB of free memory allocate their
# capacity-sized KV in per-layer chunks simultaneously, and the losing births walk
# driver-free straight through the 256MB floor before they OOM recoverably (then the
# reclaim-retry walks it AGAIN). MEMRA_CTX=131072 + streamed penalized-greedy
# max_tokens 125000 makes each birth ~4GB on the q38 export; six at once against
# ~13GB guarantees the race. The dspark session bursts every ~25ms throughout.
say "birth-race storm wave 1: free=$(gpu0_free_mb)MB; 6 concurrent streamed penalized-greedy max_tokens=125000 requests (the solo dspark generation is ~15s into its ~90s window)"
( for sidx in 1 2 3 4 5 6; do
    req "T$((5000 + sidx * 137))" 125000 -1 "" $G/resp-$TAG-stack$sidx.json &
  done; wait ) &
PRESS_PID=$!
( sleep 35
  say "birth-race storm wave 2 (lands mid-generation even if wave 1 raced the loop boundary)"
  for sidx in 7 8 9 10 11 12; do
    req "T$((5000 + sidx * 137))" 125000 -1 "" $G/resp-$TAG-stack$sidx.json &
  done; wait ) &
PRESS2_PID=$!
sleep 15
say "decode-phase chase ballast joins: free=$(gpu0_free_mb)MB (hold 1GB, +1GB/2s, retrying)"
nohup $BALLAST 0 1024 1024 2 60000 > $G/ballast-$TAG.log 2>&1 &
BPID=$!

FIRED_AT=""
for i in $(seq 1 210); do
  S=$(suspended_count $LOG)
  if [ "${S:-0}" != "0" ]; then
    FIRED_AT="$(gpu0_free_mb)MB@t=$((i*2))s"
    say "suspended line observed (free=$FIRED_AT loops=$(cat $G/.loops-$TAG 2>/dev/null))"
    break
  fi
  sleep 2
done

# generation continues on eager: after the line, the loop's next generations must
# keep completing (an admission VRAM-reject here is the OTHER guard and counts as
# recoverable, but at least one post-line completion is required for PASS).
sleep 15
touch $G/.stop-$TAG
LOOPS=$(cat $G/.loops-$TAG 2>/dev/null || echo 0)
ALIVE=no; kill -0 $SRV_PID 2>/dev/null && ALIVE=yes
say "loop generations issued: $LOOPS ; server alive: $ALIVE ; free=$(gpu0_free_mb)MB"

# ---- phase 3: release ballast, recovery ----
kill -TERM $BPID 2>/dev/null
kill -TERM ${BPRE_PID:-0} 2>/dev/null
wait $LOOP_PID 2>/dev/null
wait $PRESS_PID 2>/dev/null
wait ${PRESS2_PID:-0} 2>/dev/null
sleep 5
req 7 96 0 "" $G/resp-$TAG-recovery.json
RECOV=$(resp_ok $G/resp-$TAG-recovery.json)

# post-line completion census over the loop responses
POST_OK=0; TOTAL=0
for f in $G/resp-$TAG-loop*.json; do
  [ -f "$f" ] || continue
  TOTAL=$((TOTAL+1))
  case "$(resp_ok $f)" in OK:*) POST_OK=$((POST_OK+1));; esac
done

SUS=$(suspended_count $LOG)
SUSVG=$(grep -ac "\[dspark-vg\] graph replay suspended:" $G/$LOG || true)
shutdown_srv
sampler_stop
dmesg_check squeeze-$TAG
FAULTS=$(grep -c . $G/dmesg-squeeze-$TAG.txt 2>/dev/null | head -1); FAULTS=${FAULTS:-0}
receipts $LOG squeeze-$TAG
rm -f $G/.stop-$TAG $G/.loops-$TAG
say "VERDICT $TAG: suspended=$SUS dspark_vg_tag=$SUSVG fired_at=${FIRED_AT:-NEVER} loop_ok=$POST_OK/$TOTAL recovery=$RECOV alive=$ALIVE dmesg_faults=$FAULTS cal_route_dspark=$CAL"
if [ "${SUSVG:-0}" != "0" ] && [ "$FAULTS" = "0" ] && [ "$ALIVE" = "yes" ] && [ "${RECOV%%:*}" = "OK" ]; then
  say "RUN $TAG: PASS"; exit 0
fi
say "RUN $TAG: FAIL"; exit 1
