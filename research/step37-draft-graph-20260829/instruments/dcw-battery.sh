#!/bin/bash
# step37 draft-graph dcw door battery (lane step37-draft-graph-20260829), v2.
#
# v2 exists because v1 broke a co-lane: it rebuilt the SHARED checkout's binary in place
# (the sq lane pins that path's md5) and held the GPU lock for the whole battery. This
# version runs PRIVATE binaries (/root/dcw-server, /root/dcw-run-spec, saved from the
# lane build at dfe718016) and takes the lock PER CELL (the sq lane's per-block pattern),
# so the three lanes sharing this box interleave instead of starving.
#
# GATE A: run-spec greedy identity, K=1..8, real prompt (curve-0400), three arms:
#   off        today's serving (eager mtp_step35_attn + capture refusal WARN)
#   dcw-eager  MEMRA_STEP35_DRAFT_DCW=1 MEMRA_SPEC_NOGRAPH=1 (the dcw launcher, eager chain)
#   dcw-graph  MEMRA_STEP35_DRAFT_DCW=1 (captured chain; WARN must be ABSENT)
#   Bars: SELF-CONSISTENCY PASS in every arm at every K; per-K acceptance IDENTICAL between
#   dcw-eager and dcw-graph (same launcher, same bucket: drafts bit-identical by
#   construction); the [mtp-geom] arm receipt matches the arm in BOTH directions.
#   Plus one serving-policy twin (K=3 PMIN=0.5 PMIN0=1) per arm.
#
# CELL C: vendor-default SAMPLED decode at the serving env, arms door-off / door-on,
#   interleaved x5 alternating within one stretch of box life (clock-drift law). Engagement
#   from usage.spec in the RESPONSE BODY per arm; thinking-model hygiene (reasoning+content
#   hashed, empty completion INVALID, loop-shaped rows flagged and excluded) lives in the
#   probe (dcw-spec-probe.py, the s37h instrument verbatim).
set -u
OUT=/root/dcw-battery.txt; : > $OUT
BIN=/root/dcw-server
RS=/root/dcw-run-spec
MODEL=/root/models/step37-flash-nvfp4
LOCKWAIT=14400

lockrun() { # run "$@" while holding the box GPU lock; release between cells
  (
    exec 9>/root/gemmprime.lock
    flock -w $LOCKWAIT 9 || { echo "LOCK_TIMEOUT for: $*" >> $OUT; exit 99; }
    "$@"
  )
}

{
  echo "date=$(date -Is)"
  echo "lane_branch_tip=$(git -C /home/ubuntu/memra log -1 --format='%h %s' lane/step37-draft-dcw-20260829)"
  echo "server bin=$BIN md5=$(md5sum $BIN | cut -d' ' -f1)"
  # BINARY FINGERPRINT from strings, never cargo's Finished line.
  echo "server strings dcw_flag=$(strings -a $BIN | grep -c MEMRA_STEP35_DRAFT_DCW) arm_dcw=$(strings -a $BIN | grep -c 'arm=dcw')"
  echo "run-spec bin=$RS md5=$(md5sum $RS | cut -d' ' -f1) dcw_flag=$(strings -a $RS | grep -c MEMRA_STEP35_DRAFT_DCW)"
} >> $OUT

BASE=$(grep "^ENVV=" /root/agentic8.sh | sed 's/^ENVV=//; s/^"//; s/"$//')
python3 /root/dcw-mkprompt.py >> $OUT 2>&1

runspec_cell() { # $1=cell $2=arm $3=door-env $4=policy-env
  local LOG=/root/dcw-runspec-$1-$2.log
  env $BASE MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=3 MEMRA_CHAT=1 MEMRA_NGEN=160 \
    MEMRA_PROMPT_FILE=/root/dcw-0400.prompt $3 $4 \
    timeout 3600 $RS $MODEL > $LOG 2>&1
  local RC=$?
  {
    echo "cell=$1 arm=$2 rc=$RC"
    grep -aE "\[generate_spec K=|acceptance:|SELF-CONSISTENCY" $LOG | sed 's/^/  /'
    echo "  geom_receipt: $(grep -a '\[mtp-geom\]' $LOG | head -1)"
    echo "  capture_warn_count=$(grep -ac 'draft-graph capture failed' $LOG)"
    echo "  illegal=$(grep -aic 'ILLEGAL' $LOG) sentinel87=$(grep -ac '#87' $LOG)"
  } >> $OUT
}

echo "=== GATE A: run-spec greedy identity (curve-0400, NGEN=160) ===" >> $OUT
for CELL in sweep policy; do
 for ARM in off dcw-eager dcw-graph; do
  case $ARM in
    off)       DOOR="" ;;
    dcw-eager) DOOR="MEMRA_STEP35_DRAFT_DCW=1 MEMRA_SPEC_NOGRAPH=1" ;;
    dcw-graph) DOOR="MEMRA_STEP35_DRAFT_DCW=1" ;;
  esac
  case $CELL in
    sweep)  POL="" ;;  # no MEMRA_SPEC_K -> full K=1..8 sweep
    policy) POL="MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1" ;;
  esac
  lockrun runspec_cell $CELL $ARM "$DOOR" "$POL"
 done
done

boot_block() { # $1=arm $2=door-env $3=round  -- boot, probe, receipts, kill (one lock hold)
  local ARM=$1 DOOR=$2 RND=$3
  local LOG=/root/dcw-c-$ARM-$RND.log
  env $BASE $COMMON $SPECPOL $DOOR MEMRA_MODELS="step37=$MODEL" \
    MEMRA_ADDR=127.0.0.1:$P nohup $BIN > $LOG 2>&1 &
  local SRVPID=$!
  local up=0
  for hw in $(seq 1 600); do
    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$P/health 2>/dev/null | grep -q 200 && { up=1; break; }
    kill -0 $SRVPID 2>/dev/null || break
    sleep 5
  done
  if [ "$up" = "1" ]; then
    ARM=$ARM P=$P RND=$RND python3 /root/dcw-spec-probe.py >> $OUT 2>&1
    # Boot receipt pair, BOTH directions, from the SERVER log per boot:
    echo "      geom=$(grep -a '\[mtp-geom\]' $LOG | head -1 | cut -c1-90)" >> $OUT
    echo "      capture_warn=$(grep -ac 'draft-graph capture failed' $LOG) illegal=$(grep -aic ILLEGAL $LOG) sentinel87=$(grep -ac '#87' $LOG)" >> $OUT
  else
    echo "rnd=$RND arm=$ARM booted=NO - CELL INVALID" >> $OUT
    tail -5 $LOG >> $OUT
  fi
  # Kill OUR server only (a shared box runs other lanes' processes).
  kill -TERM $SRVPID 2>/dev/null; sleep 15; kill -KILL $SRVPID 2>/dev/null; sleep 5
}

echo "=== CELL C: vendor-default sampled, serving env, interleaved x5 ===" >> $OUT
SPECPOL="MEMRA_SERVE_SPEC=1 MEMRA_SPEC_K=3 MEMRA_MTP_HEADS=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1"
COMMON="MEMRA_LOAD_MTP=1 MEMRA_CTX=262144"
P=19310
for RND in 1 2 3 4 5; do
 for ARM in dcw-off dcw-on; do
  case $ARM in
    dcw-off) DOOR="" ;;
    dcw-on)  DOOR="MEMRA_STEP35_DRAFT_DCW=1" ;;
  esac
  lockrun boot_block $ARM "$DOOR" $RND
 done
done
echo "DCW-BATTERY-DONE" >> $OUT
