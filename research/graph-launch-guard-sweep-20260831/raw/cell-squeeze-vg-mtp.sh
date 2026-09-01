#!/bin/bash
# CELL: guard-fires squeeze on the VERIFY-GRAPH route via the MTP spec caller
# (lane/graph-launch-guard-sweep-20260831).
#
# ROUTE STATEMENT (exact, for the report): this cell exercises spec.rs
# run_full/run_segment (sites 3140/3299) through decode_step_t_core_vg -> the guarded
# qwen35_verify_tparallel, i.e. the MTP-route verify-graph door
# (MEMRA_SPEC_VERIFY_GRAPH=1 on the MTP-carrying q38 artifact) - the ornith-class
# serve program. The dflash-SERVE caller of the same two sites is NOT reachable with
# on-box artifacts: runs 1-20 measured zero vg captures on the q38+DFlash2 shape and
# dflash.rs's serve burst confirms `deferred` (the only arm that passes vgraphs) is
# never set on the DFlash2 branch - only markov/plain-chain drafters arm it. Both
# callers run the SAME guard in the SAME shared function.
#
# Squeeze mechanics (adopted from step37 phaseF receipts on this box): pre-ballast so
# the server boots into ~13GB free, teeth door MEMRA_ADMIT_RESERVE_MB=16, a solo MTP
# spec generation bursting throughout, then a race of concurrent multi-GB capacity
# births + the MTP capture appetite walks DRIVER free (mem_get_info, NOT nvidia-smi -
# the run-19 defer receipts showed driver-free 86MB while nvidia-smi read 717MB)
# through the 256MB floor; the guard flips every captured-graph arm to eager with the
# grep-stable `graph replay suspended:` line.
# FALSE-POSITIVE check: any suspended line at healthy headroom kills the run.
# Usage: cell-squeeze-vg-mtp.sh <run-tag>
set -u
. /home/ubuntu/guard-lane/gl-lib.sh
SERVE_ENV="$SERVE_ENV_MTP"
TAG=${1:?run tag}
LOG=serve-vgmtp-$TAG.log
resp_ok() { python3 -c "import json,sys;d=json.load(open('$1'));t=d.get('text') or (d.get('choices') or [{}])[0].get('text');print('OK:'+str(len(t)) if t else 'FAIL:'+str(d)[:120])" 2>/dev/null || echo PARSE_FAIL; }

say "=== VG-MTP SQUEEZE RUN $TAG (bin=lane) ==="
gpu_empty || { say "REFUSING: GPU not empty / server alive"; exit 2; }
dmesg_mark
sampler_start $G/free-vgmtp-$TAG.csv
nohup $BALLAST 0 64000 > $G/ballast-pre-$TAG.log 2>&1 &
BPRE_PID=$!
sleep 8
say "pre-ballast holding: free=$(gpu0_free_mb)MB"
boot "$BINLANE" "MEMRA_ADMIT_RESERVE_MB=16 MEMRA_GRAPH_CENSUS=1" "$LOG" || { sampler_stop; kill -9 $BPRE_PID 2>/dev/null; exit 1; }

# ---- phase 1: engagement + false-positive check ----
req 0 96 0 "" $G/resp-$TAG-p1a.json
req 1 96 0 "" $G/resp-$TAG-p1b.json
ENG=$(grep -ac 'verify-graph pool ENGAGED' $G/$LOG || true)
CEN=$(grep -ac 'dspark-vg-census' $G/$LOG || true)
SACC=$(grep -ac 'spec-acc' $G/$LOG || true)
S=$(suspended_count $LOG)
if [ "${S:-0}" != "0" ]; then
  say "FALSE-POSITIVE: $S suspended line(s) at healthy headroom; killing run"
  shutdown_srv; sampler_stop; dmesg_check vgmtp-$TAG; exit 9
fi
say "phase1 clean: suspended=0 vg_pool_engaged=$ENG vg_captures=$CEN spec_acc=$SACC free=$(gpu0_free_mb)MB"
if [ "${ENG:-0}" = "0" ] || [ "${CEN:-0}" = "0" ]; then
  say "ROUTE NOT REPLAYING (pool=$ENG captures=$CEN): refusing to claim coverage"
  shutdown_srv; sampler_stop; kill -9 $BPRE_PID 2>/dev/null; exit 7
fi

# ---- phase 2: solo MTP spec generations + birth-race storm ----
rm -f $G/.stop-$TAG
( n=0
  while [ ! -f $G/.stop-$TAG ]; do
    n=$((n+1))
    req "u$n" 4900 0 "" $G/resp-$TAG-loop$n.json
    echo $n > $G/.loops-$TAG
    sleep 1
  done ) &
LOOP_PID=$!
sleep 8
if [ -s $G/resp-$TAG-loop1.json ]; then
  case "$(resp_ok $G/resp-$TAG-loop1.json)" in
    OK:*) : ;;
    *) say "LOOP REQUEST REFUSED/BROKEN: $(head -c 200 $G/resp-$TAG-loop1.json)"; touch $G/.stop-$TAG; shutdown_srv; sampler_stop; exit 6 ;;
  esac
fi

say "birth-race storm wave 1: free=$(gpu0_free_mb)MB; 6 concurrent streamed penalized-greedy max_tokens=125000 requests"
( for sidx in 1 2 3 4 5 6; do
    req "T$((5000 + sidx * 137))" 125000 -1 "" $G/resp-$TAG-stack$sidx.json &
  done; wait ) &
PRESS_PID=$!
( sleep 35
  say "birth-race storm wave 2"
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
    FIRED_AT="$(gpu0_free_mb)MB(nvidia-smi)@t=$((i*2))s"
    say "suspended line observed ($FIRED_AT loops=$(cat $G/.loops-$TAG 2>/dev/null))"
    break
  fi
  sleep 2
done

sleep 15
touch $G/.stop-$TAG
LOOPS=$(cat $G/.loops-$TAG 2>/dev/null || echo 0)
ALIVE=no; kill -0 $SRV_PID 2>/dev/null && ALIVE=yes
say "loop generations issued: $LOOPS ; server alive: $ALIVE"

# ---- phase 3: release ballast, recovery ----
kill -TERM $BPID 2>/dev/null
kill -TERM ${BPRE_PID:-0} 2>/dev/null
wait $LOOP_PID 2>/dev/null
wait $PRESS_PID 2>/dev/null
wait ${PRESS2_PID:-0} 2>/dev/null
sleep 5
req 7 96 0 "" $G/resp-$TAG-recovery.json
RECOV=$(resp_ok $G/resp-$TAG-recovery.json)

POST_OK=0; TOTAL=0
for f in $G/resp-$TAG-loop*.json; do
  [ -f "$f" ] || continue
  TOTAL=$((TOTAL+1))
  case "$(resp_ok $f)" in OK:*) POST_OK=$((POST_OK+1));; esac
done

SUS=$(suspended_count $LOG)
shutdown_srv
sampler_stop
dmesg_check vgmtp-$TAG
FAULTS=$(grep -c . $G/dmesg-vgmtp-$TAG.txt 2>/dev/null | head -1); FAULTS=${FAULTS:-0}
{
  echo "--- receipts vgmtp-$TAG"
  echo "vg_pool_engaged: $(grep -ac 'verify-graph pool ENGAGED' $G/$LOG)"
  echo "vg_captures: $(grep -ac 'dspark-vg-census' $G/$LOG)"
  echo "suspended_total: $(grep -ac 'graph replay suspended:' $G/$LOG)"
  grep -a 'graph replay suspended:' $G/$LOG | head -3
  echo "oom_lines: $(grep -ac 'CUDA_ERROR_OUT_OF_MEMORY' $G/$LOG)"
  echo "illegal=$(grep -aic 'ILLEGAL' $G/$LOG) sentinel87=$(grep -ac '#87' $G/$LOG) panic=$(grep -aic 'panic' $G/$LOG)"
} | tee -a $OUT
rm -f $G/.stop-$TAG $G/.loops-$TAG
say "VERDICT $TAG: suspended=$SUS fired_at=${FIRED_AT:-NEVER} vg_captures=$CEN loop_ok=$POST_OK/$TOTAL recovery=$RECOV alive=$ALIVE dmesg_faults=$FAULTS"
if [ "${SUS:-0}" != "0" ] && [ "$FAULTS" = "0" ] && [ "$ALIVE" = "yes" ] && [ "${RECOV%%:*}" = "OK" ]; then
  say "RUN $TAG: PASS"; exit 0
fi
say "RUN $TAG: FAIL"; exit 1
