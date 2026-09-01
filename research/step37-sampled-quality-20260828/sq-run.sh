#!/bin/bash
# Sampled-quality cell runner. One boot per arm-block, interleaved cycles
# (cold+walk on door=0, gemm on door=1, boot order alternating by cycle parity),
# cold rows always the first requests of their boot. Resume-safe: a row whose
# gen file exists is skipped, transcript is built once.
set -u
SQ=/root/sq
mkdir -p $SQ/gen
# GPU lock: taken PER BOOT-BLOCK (subshell around boot+rows+down), not for the whole
# battery: rb37 and dcw lanes share this box's lock (discovered 2026-08-28 ~22:15Z),
# and a multi-hour hold would starve them. Each block is self-contained.
LOCKWAIT=${SQ_LOCKWAIT:-21600}
# 2026-08-28 ~22:20Z: a co-tenant lane rebuilt /home/ubuntu/memra at another tip
# (dfe718016, md5 e42d389d...); the md5 gate caught it. This cell now builds and runs
# its OWN binary from the pinned tip in an isolated worktree.
BIN=/home/ubuntu/memra-sq/target/release/memra-server
# Rebuild of the SAME pinned tip in the isolated worktree (the tasking's original
# binary f45c3623... was overwritten by the co-tenant rebuild): md5 09fe2d67...,
# markers verified, cuda-linked, size within 120 bytes of the reference build.
EXPECT_MD5=${SQ_EXPECT_MD5:-09fe2d670d82931248d4b0733898e6f4}
EXPECT_TIP=8695bdef4
MD5=$(md5sum $BIN | cut -d' ' -f1)
TIP=$(cd /home/ubuntu/memra-sq && git log -1 --format=%h)
echo "RESULTS HEADER: bin=$BIN md5=$MD5 expect=$EXPECT_MD5 tip=$TIP expect_tip=$EXPECT_TIP date=$(date -u +%FT%TZ)"
[ "$MD5" = "$EXPECT_MD5" ] || { echo "MD5_MISMATCH"; exit 2; }
[ "$TIP" = "$EXPECT_TIP" ] || { echo "TIP_MISMATCH"; exit 2; }
for M in MEMRA_ROWS_TAB_RESTAGE MEMRA_STEP_GEMM_PRIME_SUFFIX ckpt-bounds-take-v2; do
  N=$(strings -a $BIN | grep -c "$M" || true)
  echo "fix_marker $M count=$N"
  [ "$N" -ge 1 ] || { echo "MARKER_MISSING $M"; exit 2; }
done
BASE=$(grep "^ENVV=" /root/agentic8.sh | sed 's/^ENVV=//; s/^"//; s/"$//')
PORT=18902
GATES="MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1"

SERVER_PID=""
CUR_LOG=""

boot() { # $1=door $2=bootid
  CUR_LOG=$SQ/server-$2.log
  local T0=$(date +%s)
  echo "[boot $2] door=$1 log=$CUR_LOG"
  env $BASE $GATES MEMRA_STEP_GEMM_PRIME_SUFFIX=$1 RUST_BACKTRACE=1 \
    MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" MEMRA_ADDR=127.0.0.1:$PORT \
    nohup $BIN > $CUR_LOG 2>&1 &
  SERVER_PID=$!
  # NB: the wait counter must NOT be `i`; the cycle loop uses `i` and bash for-loop
  # variables are global (attempt-2 bug: samples got numbered by the boot wait count).
  for hw in $(seq 1 540); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/health 2>/dev/null)
    [ "$CODE" = "200" ] && { echo "[boot $2] HEALTH=200 boot_seconds=$(( $(date +%s) - T0 ))"; return 0; }
    kill -0 $SERVER_PID 2>/dev/null || { echo "[boot $2] SERVER_DIED"; tail -30 $CUR_LOG; return 1; }
    sleep 5
  done
  echo "[boot $2] BOOT_TIMEOUT"; return 1
}

down() { # $1=bootid
  local ILL=$(grep -ac ILLEGAL $CUR_LOG)
  local H87=$(grep -ac '#87' $CUR_LOG)
  local PAN=$(grep -ac "panicked at" $CUR_LOG)
  echo "[down $1] ILLEGAL=$ILL hash87=$H87 panics=$PAN"
  if [ "$ILL" != "0" ] || [ "$H87" != "0" ] || [ "$PAN" != "0" ]; then
    echo "FAULT_FOUND boot=$1 ILLEGAL=$ILL hash87=$H87 panics=$PAN" | tee -a $SQ/FAULT
  fi
  kill -TERM $SERVER_PID 2>/dev/null; sleep 12; kill -KILL $SERVER_PID 2>/dev/null; sleep 3
  pgrep -x memra-server >/dev/null && echo "[down $1] SERVER_STILL_UP" || echo "[down $1] SERVER_GONE"
}

row() { # $1=arm $2=turn $3=sample $4=bootid
  if [ -f $SQ/gen/$1-t$2-s$3.json ]; then echo "[row] SKIP $1 t$2 s$3 (banked)"; return 0; fi
  echo "[row] $1 t$2 s$3 boot=$4 $(date -u +%T)"
  # Warm rows replay the true sequential shape; the grow panic (FINDINGS.md) is dodged
  # by pre-sizing the session on the first prefix request (driver PRESIZE), so no
  # parked-session grow ever happens. SQ_T8_SHAPE=onegrow remains as a fallback seam.
  local SHAPE=seq
  if [ "$2" = "8" ] && [ "$1" != "cold" ]; then SHAPE=${SQ_T8_SHAPE:-seq}; fi
  env P=$PORT LOG=$CUR_LOG ARM=$1 TURN=$2 SAMPLE=$3 BOOT_ID=$4 BIN_MD5=$MD5 SQ_WARM_SHAPE=$SHAPE \
    python3 $SQ/sq-drive.py row || echo "[row] DRIVER_ERROR $1 t$2 s$3"
}

locked_block() { # runs "$@" (a function name + args) under the GPU lock, own subshell
  (
    exec 9>/root/gemmprime.lock
    flock -w $LOCKWAIT 9 || { echo "LOCK_TIMEOUT for $*"; exit 9; }
    "$@"
  )
}

block_tr() {
  boot 0 tr || exit 3
  env P=$PORT LOG=$CUR_LOG BOOT_ID=tr BIN_MD5=$MD5 python3 $SQ/sq-drive.py transcript
  local RC=$?
  down tr
  [ $RC -eq 0 ] || { echo "TRANSCRIPT_FAILED rc=$RC"; exit 4; }
}

block_d0() { # $1=cycle
  boot 0 c$1d0 || exit 5
  # SQ_T8_ONLY=1: turn-8 extension cycles (s9..s16) resolve the marginal walk-vs-gemm
  # quality signal at deep context; t4 was already indistinguishable at n=8.
  [ "${SQ_T8_ONLY:-0}" = "1" ] || row cold 4 $1 c$1d0
  row cold 8 $1 c$1d0
  [ "${SQ_T8_ONLY:-0}" = "1" ] || row walk 4 $1 c$1d0
  row walk 8 $1 c$1d0
  down c$1d0
}

block_d1() { # $1=cycle
  boot 1 c$1d1 || exit 5
  [ "${SQ_T8_ONLY:-0}" = "1" ] || row gemm 4 $1 c$1d1
  row gemm 8 $1 c$1d1
  down c$1d1
}

if [ ! -f $SQ/transcript.json ]; then
  locked_block block_tr || exit $?
fi

for i in $(seq ${SQ_START:-1} ${SQ_END:-8}); do
  if [ $((i % 2)) -eq 1 ]; then ORDER="0 1"; else ORDER="1 0"; fi
  for d in $ORDER; do
    if [ "$d" = "0" ]; then
      NEED=0
      for f in cold-t4-s$i cold-t8-s$i walk-t4-s$i walk-t8-s$i; do
        [ -f $SQ/gen/$f.json ] || NEED=1
      done
      [ "$NEED" = "0" ] && { echo "[cycle $i] door0 block already banked"; continue; }
      locked_block block_d0 $i || exit $?
    else
      NEED=0
      for f in gemm-t4-s$i gemm-t8-s$i; do
        [ -f $SQ/gen/$f.json ] || NEED=1
      done
      [ "$NEED" = "0" ] && { echo "[cycle $i] door1 block already banked"; continue; }
      locked_block block_d1 $i || exit $?
    fi
  done
  echo "[cycle $i] DONE $(date -u +%T)"
done
echo "SQ_RUN_DONE"
