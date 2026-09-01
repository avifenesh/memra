#!/bin/bash
# lane/step37-affinity: does session-affinity reuse succeed after the pp.rs fix, is the reused
# answer byte-identical to the cold one, and what is the TTFT win?
# ONE lock hold, ONE server at a time. Touches no other lane's files or sources.
set -u
exec 9>/root/gemmprime.lock
flock -w 43200 9 || { echo "lock timeout" >&2; exit 1; }
OUT=/root/aff-battery.txt; : > $OUT
# We hold the lock. Wait for any other lane's process to release the GPUs (teardown can lag the
# lock release), then refuse rather than co-resident-OOM someone else.
for i in $(seq 1 60); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] && break
  sleep 10
done
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$BUSY" != "0" ]; then
  echo "ABORT: $BUSY GPU compute app(s) still resident while we hold the lock" >> $OUT
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> $OUT
  exit 2
fi
BASE=$(grep "^ENVV=" /root/agentic8.sh | sed "s/^ENVV=//; s/^\"//; s/\"$//")
PORT=18700
echo "tree=$(cd /root/memra-affinity && git log -1 --format=%h) date=$(date -Is)" >> $OUT

run_arm () {
  local NAME="$1"; local BIN="$2"; local TURNS="$3"
  PORT=$((PORT+1))
  local LOG=/root/aff-$NAME.log
  echo "" >> $OUT
  echo "=== ARM $NAME bin=$BIN md5=$(md5sum $BIN | cut -c1-12) start=$(date -Is)" >> $OUT
  echo "    BINARY FINGERPRINT (strings, never cargo's Finished line):" >> $OUT
  echo "      kind-mismatch=$(strings -a $BIN | grep -c 'TP KV kind mismatch')" >> $OUT
  echo "      restore-refused=$(strings -a $BIN | grep -c 'TP KV restore refused')" >> $OUT
  echo "      drop-receipt=$(strings -a $BIN | grep -c 'stale distributed KV mirror')" >> $OUT
  env $BASE MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=1 MEMRA_CTX=262144 MEMRA_PREFILL_TICK=8192 \
    MEMRA_PP_BF16=0 MEMRA_SERVE_SPEC=0 MEMRA_DEBUG_AFFINITY=1 \
    MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" MEMRA_ADDR=127.0.0.1:$PORT \
    nohup $BIN > $LOG 2>&1 &
  local PID=$!
  local UP=0
  for i in $(seq 1 400); do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; then UP=1; break; fi
    sleep 5
  done
  if [ "$UP" != "1" ]; then
    echo "    booted=NO -- ARM INVALID" >> $OUT; tail -8 $LOG >> $OUT
    kill -TERM $PID 2>/dev/null; sleep 15; kill -KILL $PID 2>/dev/null; sleep 5; return
  fi
  echo "    booted=YES health=200" >> $OUT
  NAME=$NAME P=$PORT TURNS=$TURNS LOG=$LOG python3 /root/aff-drive.py >> $OUT 2>&1
  {
    echo "    LOG TOTALS:"
    echo "      plain-affinity rewound = $(grep -ac 'plain-affinity: rewound to' $LOG)"
    echo "      plain-affinity resume failed = $(grep -ac 'plain-affinity resume failed' $LOG)"
    echo "      spec affinity rewind failed  = $(grep -ac 'affinity rewind failed' $LOG)"
    echo "      TP KV kind mismatch          = $(grep -ac 'TP KV kind mismatch' $LOG)"
    echo "      TP KV restore refused        = $(grep -ac 'TP KV restore refused' $LOG)"
    echo "      stale mirror drop receipts   = $(grep -ac 'stale distributed KV mirror' $LOG)"
    echo "    OPERAND / RECEIPT LINES:"
    grep -aoh "plain-affinity resume failed (.\{0,400\}" $LOG | head -3
    grep -aoh "checkpoint TP KV kind mismatch at layer .\{0,400\}" $LOG | head -2
    grep -aoh "checkpoint TP KV restore refused at layer .\{0,400\}" $LOG | head -2
    grep -aoh "cleared [0-9]* stale distributed KV mirror(s) at pos [0-9]*" $LOG | sort | uniq -c | head -5
    grep -aoh "plain-affinity: rewound to .\{0,160\}" $LOG | head -8
  } >> $OUT 2>&1
  kill -TERM $PID 2>/dev/null; sleep 25; kill -KILL $PID 2>/dev/null; sleep 10
}

run_arm stock /root/memra-server.stock 6
run_arm fixed /root/memra-server.fixed 6
echo "" >> $OUT
echo "AFF-BATTERY-DONE $(date -Is)" >> $OUT
