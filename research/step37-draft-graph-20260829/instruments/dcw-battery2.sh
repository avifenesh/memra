#!/bin/bash
# step37 draft-graph dcw door battery, part 2 (lane step37-draft-graph-20260829).
#
# Battery 1's discovery: Step-3.7-Flash carries THREE MTP heads (blocks 45..47) and the
# QUALIFIED serving config runs the 3-head step-modulo prefix-replay chain, for which
# `graph_draft` is structurally OFF (`mtp_extra.is_empty()` conjunct) - no capture is even
# attempted, so battery 1's dcw-graph cells exercised the EAGER dcw arm only, and the boot
# WARN in the lane brief comes from 1-head boots. This battery:
#
# GATE A2 (MEMRA_MTP_HEADS=1): the ACTUAL capture gate.
#   off        capture attempted, step35 refusal -> the WARN must be PRESENT
#   dcw-eager  door on + MEMRA_SPEC_NOGRAPH=1 (dcw launcher, eager chain)
#   dcw-graph  door on (captured chain) -> WARN must be ABSENT, K=1..8 identity PASS,
#              acceptance per K IDENTICAL to dcw-eager (same launcher -> bit-identical drafts)
#
# CELL C2: vendor-default SAMPLED serving decision matrix, interleaved x5, four arms
#   alternating within one stretch of box life:
#   h3-off  heads=3 door off   (the QUALIFIED shipping config, 92.13 class)
#   h3-on   heads=3 door on    (dcw eager arm inside the shipping shape)
#   h1-off  heads=1 door off   (eager 1-head chain: the capture-attribution baseline)
#   h1-on   heads=1 door on    (CAPTURED 1-head chain: the candidate)
#   Engagement from usage.spec per boot; boot receipt pair (WARN/arm=) per boot.
set -u
OUT=/root/dcw-battery2.txt; : > $OUT
BIN=/root/dcw-server
RS=/root/dcw-run-spec
MODEL=/root/models/step37-flash-nvfp4
LOCKWAIT=14400

lockrun() {
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
  echo "server strings dcw_flag=$(strings -a $BIN | grep -c MEMRA_STEP35_DRAFT_DCW) arm_dcw=$(strings -a $BIN | grep -c 'arm=dcw')"
  echo "run-spec bin=$RS md5=$(md5sum $RS | cut -d' ' -f1) dcw_flag=$(strings -a $RS | grep -c MEMRA_STEP35_DRAFT_DCW)"
} >> $OUT

BASE=$(grep "^ENVV=" /root/agentic8.sh | sed 's/^ENVV=//; s/^"//; s/"$//')
[ -s /root/dcw-0400.prompt ] || python3 /root/dcw-mkprompt.py >> $OUT 2>&1

runspec_cell() { # $1=cell $2=arm $3=door-env $4=policy-env
  local LOG=/root/dcw2-runspec-$1-$2.log
  env $BASE MEMRA_LOAD_MTP=1 MEMRA_MTP_HEADS=1 MEMRA_CHAT=1 MEMRA_NGEN=160 \
    MEMRA_PROMPT_FILE=/root/dcw-0400.prompt $3 $4 \
    timeout 3600 $RS $MODEL > $LOG 2>&1
  local RC=$?
  {
    echo "cell=$1 arm=$2 rc=$RC"
    grep -aE "\[generate_spec K=|acceptance:|SELF-CONSISTENCY" $LOG | sed 's/^/  /'
    echo "  geom_receipt: $(grep -a '\[mtp-geom\]' $LOG | head -1)"
    echo "  capture_warn_count=$(grep -ac 'draft-graph capture failed' $LOG)"
    echo "  first_warn: $(grep -a 'draft-graph capture failed' $LOG | head -1 | cut -c1-160)"
    echo "  illegal=$(grep -aic 'ILLEGAL' $LOG) sentinel87=$(grep -ac '#87' $LOG)"
  } >> $OUT
}

echo "=== GATE A2: run-spec greedy identity, MEMRA_MTP_HEADS=1 (curve-0400, NGEN=160) ===" >> $OUT
for CELL in sweep policy; do
 for ARM in off dcw-eager dcw-graph; do
  case $ARM in
    off)       DOOR="" ;;
    dcw-eager) DOOR="MEMRA_STEP35_DRAFT_DCW=1 MEMRA_SPEC_NOGRAPH=1" ;;
    dcw-graph) DOOR="MEMRA_STEP35_DRAFT_DCW=1" ;;
  esac
  case $CELL in
    sweep)  POL="" ;;
    policy) POL="MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1" ;;
  esac
  lockrun runspec_cell $CELL $ARM "$DOOR" "$POL"
 done
done

boot_block() { # $1=arm $2=extra-env $3=round
  local ARM=$1 EXTRA=$2 RND=$3
  local LOG=/root/dcw2-c-$ARM-$RND.log
  env $BASE $COMMON $EXTRA MEMRA_MODELS="step37=$MODEL" \
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
    echo "      geom=$(grep -a '\[mtp-geom\]' $LOG | head -1 | cut -c1-90)" >> $OUT
    echo "      capture_warn=$(grep -ac 'draft-graph capture failed' $LOG) illegal=$(grep -aic ILLEGAL $LOG) sentinel87=$(grep -ac '#87' $LOG)" >> $OUT
  else
    echo "rnd=$RND arm=$ARM booted=NO - CELL INVALID" >> $OUT
    tail -5 $LOG >> $OUT
  fi
  kill -TERM $SRVPID 2>/dev/null; sleep 15; kill -KILL $SRVPID 2>/dev/null; sleep 5
}

echo "=== CELL C2: vendor-default sampled, four arms, interleaved x5 ===" >> $OUT
COMMON="MEMRA_LOAD_MTP=1 MEMRA_CTX=262144 MEMRA_SERVE_SPEC=1 MEMRA_SPEC_K=3 MEMRA_SPEC_PMIN=0.5 MEMRA_SPEC_PMIN0=1"
P=19315
for RND in 1 2 3 4 5; do
 for ARM in h3-off h3-on h1-off h1-on; do
  case $ARM in
    h3-off) EXTRA="MEMRA_MTP_HEADS=3" ;;
    h3-on)  EXTRA="MEMRA_MTP_HEADS=3 MEMRA_STEP35_DRAFT_DCW=1" ;;
    h1-off) EXTRA="MEMRA_MTP_HEADS=1" ;;
    h1-on)  EXTRA="MEMRA_MTP_HEADS=1 MEMRA_STEP35_DRAFT_DCW=1" ;;
  esac
  lockrun boot_block $ARM "$EXTRA" $RND
 done
done
echo "DCW-BATTERY2-DONE" >> $OUT
