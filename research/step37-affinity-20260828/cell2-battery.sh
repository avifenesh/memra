#!/bin/bash
# lane/step37-affinity cell 2: the task's same-prompt interleaved x5, plus the suffix sweep that
# tests why a GROWING conversation rewinds successfully yet wins ~nothing.
# ONE lock hold, ONE server, fixed binary only (the stock control is already banked in cell 1).
set -u
exec 9>/root/gemmprime.lock
flock -w 43200 9 || { echo "lock timeout" >&2; exit 1; }
OUT=/root/aff2-battery.txt; : > $OUT
for i in $(seq 1 60); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] && break
  sleep 10
done
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$BUSY" != "0" ]; then
  echo "ABORT: $BUSY GPU compute app(s) still resident while we hold the lock" >> $OUT
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> $OUT; exit 2
fi
BASE=$(grep "^ENVV=" /root/agentic8.sh | sed "s/^ENVV=//; s/^\"//; s/\"$//")
PORT=18800
BIN=/root/memra-server.fixed
LOG=/root/aff2-fixed.log
echo "=== ARM fixed2 bin=$BIN md5=$(md5sum $BIN | cut -c1-12) start=$(date -Is)" >> $OUT
echo "    strings: kind-mismatch=$(strings -a $BIN | grep -c 'TP KV kind mismatch') restore-refused=$(strings -a $BIN | grep -c 'TP KV restore refused') drop-receipt=$(strings -a $BIN | grep -c 'stale distributed KV mirror')" >> $OUT
env $BASE MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=1 MEMRA_CTX=262144 MEMRA_PREFILL_TICK=8192 \
  MEMRA_PP_BF16=0 MEMRA_SERVE_SPEC=0 MEMRA_DEBUG_AFFINITY=1 \
  MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" MEMRA_ADDR=127.0.0.1:$PORT \
  nohup $BIN > $LOG 2>&1 &
PID=$!
UP=0
for i in $(seq 1 400); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; then UP=1; break; fi
  sleep 5
done
if [ "$UP" != "1" ]; then
  echo "    booted=NO -- ARM INVALID" >> $OUT; tail -8 $LOG >> $OUT
  kill -TERM $PID 2>/dev/null; sleep 15; kill -KILL $PID 2>/dev/null; exit 3
fi
echo "    booted=YES health=200" >> $OUT
NAME=fixed2 P=$PORT LOG=$LOG python3 /root/aff2-drive.py >> $OUT 2>&1
{
  echo "    LOG TOTALS:"
  echo "      plain-affinity rewound       = $(grep -ac 'plain-affinity: rewound to' $LOG)"
  echo "      plain-affinity resume failed = $(grep -ac 'plain-affinity resume failed' $LOG)"
  echo "      TP KV kind mismatch          = $(grep -ac 'TP KV kind mismatch' $LOG)"
  echo "      TP KV restore refused        = $(grep -ac 'TP KV restore refused' $LOG)"
  echo "      grew parked cache            = $(grep -ac 'plain-affinity: grew parked cache' $LOG)"
  echo "      admit-oom                    = $(grep -ac 'admit-oom' $LOG)"
} >> $OUT 2>&1
kill -TERM $PID 2>/dev/null; sleep 25; kill -KILL $PID 2>/dev/null; sleep 10
echo "AFF2-DONE $(date -Is)" >> $OUT
