#!/bin/bash
# lane/gemm-suffix battery: does a CONTINUATION prime ride the batched GEMM path, is it the same
# numeric program as a cold full prime of the same bytes, and what is the TTFT win?
# ONE lock hold, ONE server at a time. Three arms of ONE binary:
#   off    MEMRA_STEP_GEMM_PRIME_SUFFIX=0   the pre-lane behaviour (suffix -> walk)
#   on     (default)                        the lane's arm (suffix -> batched GEMM prime)
#   canary MEMRA_STEP35_PRIME_BATCH_TSEND=1 the pre-fix chunk-local seq_end, which MUST break
#                                           the suffix byte-identity gate
set -u
exec 9>/root/gemmprime.lock
flock -w 14400 9 || { echo "lock timeout" >&2; exit 1; }
OUT=/root/gs-battery2.txt; : > $OUT
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
BIN=/root/memra-server.gsuffix
PORT=18900
{
  echo "tree=$(cd /root/wt-gemmsuffix && git log -1 --format=%H) date=$(date -Is)"
  echo "dirty=$(cd /root/wt-gemmsuffix && git status --porcelain | tr '\n' ' ')"
  echo "patch sha256=$(sha256sum /root/gemmsuffix.patch | cut -d' ' -f1)"
  echo "bin=$BIN md5=$(md5sum $BIN | cut -c1-12)"
  echo "BINARY FINGERPRINT (strings, never cargo's Finished line):"
  echo "  'gemm-prime] ENGAGED'              = $(strings -a $BIN | grep -c 'gemm-prime\] ENGAGED')"
  echo "  'gemm-prime] WALK'                 = $(strings -a $BIN | grep -c 'gemm-prime\] WALK')"
  echo "  'MEMRA_STEP_GEMM_PRIME_SUFFIX'     = $(strings -a $BIN | grep -c 'MEMRA_STEP_GEMM_PRIME_SUFFIX')"
  echo "  'MEMRA_STEP35_PRIME_BATCH_TSEND'   = $(strings -a $BIN | grep -c 'MEMRA_STEP35_PRIME_BATCH_TSEND')"
  echo "  'plain-affinity: rewound to'       = $(strings -a $BIN | grep -c 'plain-affinity: rewound to')"
} >> $OUT

run_arm () {
  local NAME="$1"; shift
  local EXPECT="${EXPECT:-1}"
  local EXPECT_FRESH="${EXPECT_FRESH:-1}"
  local GS_SWEEP="${GS_SWEEP:-}"
  local GS_LEGS="${GS_LEGS:-}"
  local GS_U1_WORDS="${GS_U1_WORDS:-}"
  local GS_CAL_PAD="${GS_CAL_PAD:-600}"
  PORT=$((PORT+1))
  local LOG=/root/gs2-$NAME.log
  echo "" >> $OUT
  echo "=== ARM $NAME extra_env='$*' port=$PORT start=$(date -Is)" >> $OUT
  env $BASE MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=1 MEMRA_CTX=262144 MEMRA_PREFILL_TICK=8192 \
    MEMRA_PP_BF16=0 MEMRA_SERVE_SPEC=0 MEMRA_STEP_GEMM_PRIME=1 MEMRA_DEBUG_AFFINITY=1 \
    MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" MEMRA_ADDR=127.0.0.1:$PORT \
    "$@" nohup $BIN > $LOG 2>&1 &
  local PID=$!
  local UP=0
  for i in $(seq 1 400); do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; then UP=1; break; fi
    sleep 5
  done
  if [ "$UP" != "1" ]; then
    echo "    booted=NO -- ARM INVALID" >> $OUT; tail -12 $LOG >> $OUT
    kill -TERM $PID 2>/dev/null; sleep 15; kill -KILL $PID 2>/dev/null; sleep 5; return
  fi
  echo "    booted=YES health=200" >> $OUT
  NAME=$NAME P=$PORT LOG=$LOG MAXTOK=256 EXPECT_SUFFIX_ARM=$EXPECT EXPECT_FRESH_ENG=${EXPECT_FRESH:-1} GS_SWEEP=${GS_SWEEP:-} GS_LEGS=${GS_LEGS:-} GS_U1_WORDS=${GS_U1_WORDS:-} GS_CAL_PAD=${GS_CAL_PAD:-600} python3 /root/gs-drive.py >> $OUT 2>&1
  {
    echo "    SERVER-LOG TOTALS ($LOG):"
    echo "      gemm-prime ENGAGED (all)      = $(grep -ac '\[gemm-prime\] ENGAGED' $LOG)"
    echo "      gemm-prime ENGAGED base=0     = $(grep -ao '\[gemm-prime\] ENGAGED t=[0-9]* base=0 ' $LOG | wc -l)"
    echo "      gemm-prime ENGAGED base>0     = $(grep -ao '\[gemm-prime\] ENGAGED t=[0-9]* base=[0-9]*' $LOG | grep -vc 'base=0$')"
    echo "      gemm-prime WALK (all)         = $(grep -ac '\[gemm-prime\] WALK' $LOG)"
    echo "      gemm-prime WALK base>0        = $(grep -ao '\[gemm-prime\] WALK t=[0-9]* base=[0-9]*' $LOG | grep -vc 'base=0$')"
    echo "      plain-affinity rewound        = $(grep -ac 'plain-affinity: rewound to' $LOG)"
    echo "      plain-affinity resume failed  = $(grep -ac 'plain-affinity resume failed' $LOG)"
    echo "    SAMPLE SUFFIX-ARM LINES:"
    grep -ao '\[gemm-prime\] ENGAGED t=[0-9]* base=[0-9]* seq_end=[0-9]* chunks<=[0-9]*' $LOG | grep -v 'base=0 ' | head -6
    grep -ao '\[gemm-prime\] WALK t=[0-9]* base=[0-9]* seq_end=[0-9]*' $LOG | grep -v 'base=0 ' | head -6
  } >> $OUT 2>&1
  kill -TERM $PID 2>/dev/null; sleep 25; kill -KILL $PID 2>/dev/null; sleep 10
}

# EVERY arm names the door explicitly. The committed default for the suffix door is OFF
# (unmeasured behaviour does not default ON; the flip is the receipts commit), while the
# binary under test was built before that flip -- so nothing here may lean on a default.
# WALK-ONLY DIAGNOSTIC. The whole batched prime off, so every prime -- the session's turn-1
# prefix, the cold twin, and the suffix -- runs the chunk-invariant walk. If cold-vs-rewound
# byte identity PASSES here and fails whenever the batched prime is involved, the identity
# failure is the batched prime's m-dependence and not this lane's hoist.
#
# ATTEMPT 2 GEOMETRY. Attempt 1 (banked: raw/gs-battery2-attempt1.txt) ran the battery-1
# geometry and every row died on the server's 90 s first-token deadline (TIMEOUT_MS_MAX,
# a platform cap, streaming included): the full-prompt walk prime measured ~57.5 ms/token,
# so any cold walk prime past ~1500 tokens can never answer in time. GS_U1_WORDS=480 puts
# turn 1 at ~740 tokens and the largest cold twin (s0450) at ~1230, inside the deadline
# with margin; GS_LEGS=S drops LEG F/G (unreachable at walk speed, and not this arm's
# question); GS_CAL_PAD=300 keeps the calibration probe inside the deadline too. The
# m-fork the prediction needs -- warm prefix primed at turn-1 m, cold twin primed at the
# longer turn-2 m -- is preserved at the smaller geometry.
EXPECT=0 EXPECT_FRESH=0 GS_SWEEP=s0250,s0450 GS_LEGS=S GS_U1_WORDS=480 GS_CAL_PAD=300 run_arm walkonly MEMRA_STEP_GEMM_PRIME=0

echo "" >> $OUT
echo "GS-BATTERY2-DONE $(date -Is)" >> $OUT
