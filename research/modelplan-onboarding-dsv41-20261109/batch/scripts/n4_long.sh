#!/bin/bash
# Inside one lease of four cards (a rank a card, the full resident trunk, device-push): the gate of the long-row
# selections where the candidate pool exists (layer 20 writes it, layers 24 to 36 read it). Plan n4l (batch-rows
# chat8, nine, window) twice: the long forms on every decode row (--select-long-min 0), then never
# (--select-long-min 100000000). Every job must pass in both and the hashes.txt of every job must be identical.
# usage: n4_long.sh TAG        env: DEVICES (default 0,1,2,3), LAYERS (default 40)
set -u
TAG=$1
D=$LANE
S=$D/spec
BIN=${SPEC_BIN:-$S/bin/$TAG/dsv41-spec}
DEV=${DEVICES:-0,1,2,3}
LAYERS=${LAYERS:-40}
P=$S/plans
{ grep -h "^batch-rows chat8 " $P/gate9b.plan; grep -h "^batch-rows nine " $P/gate9b.plan; grep -h "^batch-rows window " $P/gate9b.plan; } > $P/n4l.plan
t0=$(date +%s)
bad=0
for arm in always never; do
  n=0; [ $arm = never ] && n=100000000
  o=$S/out/$TAG-n4l-$arm
  mkdir -p $o
  MEMRA_TP_TRANSPORT=device-push MEMRA_FULL_PREC=1 MEMRA_DSV41_TOKEN_MAP=$D/oracle/engram_constants.token_map.i32 \
    $BIN $D/DeepSeek-V4.1-Flash-1017 --plan $P/n4l.plan --out $o --devices $DEV --layers $LAYERS --resident --no-draft \
    --select-long-min $n > $o/stdout.jsonl 2> $o/stderr.log
  rc=$?
  [ $rc -eq 0 ] || bad=1
  echo "$(date +%H:%M:%S) $arm: rc $rc $(grep -h SPEC-DONE $o/stdout.jsonl) ($(( $(date +%s) - t0 )) s)"
done
python3 $S/scripts/cmp_arms.py $S/out $TAG-n4l-never $TAG-n4l-always > $S/logs/$TAG-n4l-cmp.txt 2>&1
crc=$?
grep -h "^hashes\|jobs" $S/logs/$TAG-n4l-cmp.txt
[ $bad -eq 0 ] && [ $crc -eq 0 ] && echo "N4-LONG-DONE $TAG green" || { echo "N4-LONG RED $TAG"; exit 1; }
