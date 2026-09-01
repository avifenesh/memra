#!/bin/bash
# Sampled-quality cell runner. One boot per arm-block, interleaved cycles
# (cold+walk on door=0, gemm on door=1, boot order alternating by cycle parity),
# cold rows always the first requests of their boot. Resume-safe: a row whose
# gen file exists is skipped, transcript is built once.
set -u
SQ=/root/sq
mkdir -p $SQ/gen
exec 9>/root/gemmprime.lock
flock -w 7200 9 || { echo "LOCK_TIMEOUT"; exit 1; }
BIN=/home/ubuntu/memra/target/release/memra-server
EXPECT_MD5=f45c3623d958ca085eefd3207987812a
MD5=$(md5sum $BIN | cut -d' ' -f1)
echo "RESULTS HEADER: bin=$BIN md5=$MD5 expect=$EXPECT_MD5 tip=$(cd /home/ubuntu/memra && git log -1 --format=%h) date=$(date -u +%FT%TZ)"
[ "$MD5" = "$EXPECT_MD5" ] || { echo "MD5_MISMATCH"; exit 2; }
BASE=$(grep "^ENVV=" /root/agentic8.sh | sed 's/^ENVV=//; s/^"//; s/"$//')
PORT=18902
GATES="MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1"

SERVER_PID=""
CUR_LOG=""

boot() { # $1=door $2=bootid
  CUR_LOG=$SQ/server-$2.log
  local T0=$(date +%s)
  echo "[boot $2] door=$1 log=$CUR_LOG"
  env $BASE $GATES MEMRA_STEP_GEMM_PRIME_SUFFIX=$1 \
    MEMRA_MODELS="step37=/root/models/step37-flash-nvfp4" MEMRA_ADDR=127.0.0.1:$PORT \
    nohup $BIN > $CUR_LOG 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 540); do
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
  echo "[down $1] ILLEGAL=$ILL hash87=$H87"
  if [ "$ILL" != "0" ] || [ "$H87" != "0" ]; then
    echo "FAULT_FOUND boot=$1 ILLEGAL=$ILL hash87=$H87" | tee -a $SQ/FAULT
  fi
  kill -TERM $SERVER_PID 2>/dev/null; sleep 12; kill -KILL $SERVER_PID 2>/dev/null; sleep 3
  pgrep -x memra-server >/dev/null && echo "[down $1] SERVER_STILL_UP" || echo "[down $1] SERVER_GONE"
}

row() { # $1=arm $2=turn $3=sample $4=bootid
  if [ -f $SQ/gen/$1-t$2-s$3.json ]; then echo "[row] SKIP $1 t$2 s$3 (banked)"; return 0; fi
  echo "[row] $1 t$2 s$3 boot=$4 $(date -u +%T)"
  env P=$PORT LOG=$CUR_LOG ARM=$1 TURN=$2 SAMPLE=$3 BOOT_ID=$4 BIN_MD5=$MD5 \
    python3 $SQ/sq-drive.py row || echo "[row] DRIVER_ERROR $1 t$2 s$3"
}

if [ ! -f $SQ/transcript.json ]; then
  boot 0 tr || exit 3
  env P=$PORT LOG=$CUR_LOG BOOT_ID=tr BIN_MD5=$MD5 python3 $SQ/sq-drive.py transcript
  RC=$?
  down tr
  [ $RC -eq 0 ] || { echo "TRANSCRIPT_FAILED rc=$RC"; exit 4; }
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
      boot 0 c${i}d0 || exit 5
      row cold 4 $i c${i}d0
      row cold 8 $i c${i}d0
      row walk 4 $i c${i}d0
      row walk 8 $i c${i}d0
      down c${i}d0
    else
      NEED=0
      for f in gemm-t4-s$i gemm-t8-s$i; do
        [ -f $SQ/gen/$f.json ] || NEED=1
      done
      [ "$NEED" = "0" ] && { echo "[cycle $i] door1 block already banked"; continue; }
      boot 1 c${i}d1 || exit 5
      row gemm 4 $i c${i}d1
      row gemm 8 $i c${i}d1
      down c${i}d1
    fi
  done
  echo "[cycle $i] DONE $(date -u +%T)"
done
echo "SQ_RUN_DONE"
