#!/bin/bash
# Inside a one-card lease: start memra-server on the deepseek-v4.1 checkpoint, wait for readiness (bounded), run the chat
# gate, stop the server and wait for it to leave.
. "${LANE_ENV:?LANE_ENV names the machine file that sets ROOT GPU_UUID BOX_LOCK QUIET_FILE CORES GPU_RUN BUILD}"
set -u
D=$ROOT
S=$D/serve
T=$D/target-serve/release
TM=$D/oracle/engram_constants.token_map.i32
ADDR=127.0.0.1:18341
KEY=memra-dsv41-serve-gate
cd $D
MEMRA_MODELS=v41=DeepSeek-V4.1-Flash-1017 MEMRA_ADDR=$ADDR MEMRA_API_KEY=$KEY MEMRA_DSV41_TOKEN_MAP=$TM \
  $T/memra-server > $S/logs/serve-chat.log 2>&1 &
SRV=$!
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 $SRV 2>/dev/null; then echo "server exited before readiness"; break; fi
  if curl -fs -o /dev/null http://$ADDR/readyz; then ready=1; break; fi
  sleep 2
done
rc=1
if [ $ready = 1 ]; then
  echo "server ready after $SECONDS s"
  python3 $S/scripts/chat_gate.py $ADDR $KEY v41 $D/oracle/v2_cases.json $D/dec $D/dec/ids $S/chat-gate.json \
    > $S/logs/chat-gate.log 2>&1
  rc=$?
fi
kill -TERM $SRV 2>/dev/null
for _ in $(seq 1 120); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -KILL $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo "chat gate rc $rc"
exit $rc
