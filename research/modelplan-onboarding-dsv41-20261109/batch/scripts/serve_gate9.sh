#!/bin/bash
# Inside a one-card lease: the served gates on a 9-layer trunk with every weight on the card.
#  1. native greedy references from dsv41-run on the same 9 layers (every call alone through each block)
#  2. memra-server with MEMRA_DSV41_TRUNK_LAYERS=9 and the arm's environment, the serve lane's chat gate against it
#     (render, solo == native, five at once == solo, stream, seeded sampled solo and beside four, stops, penalties)
# usage: serve_gate9.sh OUT_NAME [ENV=VALUE ...]
set -u
OUT=$1; shift
D=$LANE
S=$D/spec
T=${TBIN:-$D/target-spec/release}
TM=$D/oracle/engram_constants.token_map.i32
O=$S/out/$OUT
ADDR=127.0.0.1:${PORT:-18362}
KEY=memra-dsv41-spec-gate
mkdir -p $O
cd $D
sha256sum $T/memra-server $T/dsv41-run > $O/sha256.txt
REF=$S/greedy9
if [ ! -f $REF/greedy/needle1200/gen.i32 ]; then
  mkdir -p $REF/greedy
  : > $REF/cases.txt
  for c in capital gsm multiturn sys_json hebrew needle needle1200; do echo "$c $D/dec/ids/$c.prompt.i32" >> $REF/cases.txt; done
  MEMRA_FULL_PREC=1 $T/dsv41-run $D/DeepSeek-V4.1-Flash-1017 --token-map $TM --cases $REF/cases.txt --gen 200 \
    --layers 9 --resident 9 --preload --out $REF/greedy > $REF/native.log 2>&1
  echo "native greedy rc $? $(tail -n 1 $REF/native.log | cut -c1-200)"
fi
printf '%s\n' "$@" > $O/arm.env
env MEMRA_MODELS=v41=DeepSeek-V4.1-Flash-1017 MEMRA_ADDR=$ADDR MEMRA_API_KEY=$KEY MEMRA_DSV41_TOKEN_MAP=$TM \
  MEMRA_DSV41_TRUNK_LAYERS=9 MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 "$@" $T/memra-server > $O/server.log 2>&1 &
SRV=$!
ready=0
for _ in $(seq 1 300); do
  if ! kill -0 $SRV 2>/dev/null; then echo "server exited before readiness"; break; fi
  if curl -fs -o /dev/null http://$ADDR/readyz; then ready=1; break; fi
  sleep 2
done
rc=1
if [ $ready = 1 ]; then
  echo "server ready after $SECONDS s"
  python3 $D/serve/scripts/chat_gate.py $ADDR $KEY v41 $D/oracle/v2_cases.json $REF $D/dec/ids $O/chat-gate.json \
    > $O/chat-gate.log 2>&1
  rc=$?
fi
kill -TERM $SRV 2>/dev/null
for _ in $(seq 1 120); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -KILL $SRV 2>/dev/null
wait $SRV 2>/dev/null
grep -E "dsv41-serve" $O/server.log | head -n 6 | cut -c1-260
grep -c "dsv41-step" $O/server.log
grep -E "decode_rows=[2-9]" $O/server.log | head -n 3 | cut -c1-200
tail -n 14 $O/chat-gate.log | cut -c1-300
echo "serve_gate9 $OUT rc $rc"
exit $rc
