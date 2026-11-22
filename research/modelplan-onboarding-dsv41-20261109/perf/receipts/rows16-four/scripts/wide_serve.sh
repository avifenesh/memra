#!/bin/bash
# Start memra-server (TBIN) on the deepseek-v4.1 checkpoint with the given environment, run wide_gate.py on the 16
# prompts of n4b.plan's wide16 job (64 tokens each), stop the server and wait for it to leave.
# usage: wide_serve.sh OUT_DIR DEVICES [ENV=VALUE ...]     env: TBIN, PORT (default 18393)
set -u
O=$1; DEV=$2; shift 2
D=/home/access/dsv41
S=$D/spec
ADDR=127.0.0.1:${PORT:-18393}
mkdir -p $O
cd $D
printf '%s\n' "$@" > $O/arm.env
sha256sum $TBIN/memra-server > $O/sha256.txt
env MEMRA_MODELS=v41=DeepSeek-V4.1-Flash-1017 MEMRA_ADDR=$ADDR MEMRA_DSV41_TOKEN_MAP=$D/oracle/engram_constants.token_map.i32 \
  MEMRA_COMPAT=openai MEMRA_DSV41_DEVICES=$DEV MEMRA_MAX_SESSIONS=16 MEMRA_CTX=8192 "$@" $TBIN/memra-server > $O/server.log 2>&1 &
SRV=$!
ready=0
for _ in $(seq 1 300); do
  kill -0 $SRV 2>/dev/null || { echo "server exited before readiness"; break; }
  curl -fs -o /dev/null http://$ADDR/readyz && { ready=1; break; }
  sleep 2
done
rc=1
if [ $ready = 1 ]; then
  P=$(grep '^batch-rows wide16 ' $S/plans/n4b.plan | cut -d' ' -f3- | tr ' ' '\n' | awk 'NR % 2 == 1' | sed "s|\$LANE|$D|")
  python3 $S/scripts/wide_gate.py $ADDR 64 $O/wide-gate.json $P
  rc=$?
fi
kill $SRV 2>/dev/null
for _ in $(seq 1 60); do kill -0 $SRV 2>/dev/null || break; sleep 1; done
kill -9 $SRV 2>/dev/null
echo "wide_serve rc $rc"
exit $rc
