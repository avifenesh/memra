#!/bin/bash
# The gate of the long-row selections (Ops::topk_long / candidates_long in the solo and batched index paths) on one
# card, 9-layer resident trunk. Plan long9: verify-rows at 41k, 64k and 73.5k tokens of context (at 73.5k a ratio-2
# layer holds 36,760 keys, past the top-k threshold: the one-token arm takes the long form, the block arm the
# one-block form, and the job compares them row by row), and batch-rows of a 73.5k and an 18k sequence beside short
# ones. Three runs: the default thresholds, the long forms never (--select-long-min 100000000), the long forms on
# every decode row (--select-long-min 0); hashes.txt of every job must be identical across the three.
# usage: long_gate.sh TAG CARD
set -u
TAG=$1; CARD=$2
S=$LANE/spec
I=$LANE/dec/ids
F=$S/ids/cross128.force.i32
{
  echo "verify-rows decode32k_he_summary $I/decode32k_he_summary.prompt.i32 $I/decode32k_he_summary.gen.i32"
  echo "verify-rows decode64k_en_continue $I/decode64k_en_continue.prompt.i32 $I/decode64k_en_continue.gen.i32"
  echo "verify-rows needle4800 $I/needle4800.prompt.i32 $F"
  echo "batch-rows longmix $I/gsm.prompt.i32 $I/gsm.gen.i32 $I/needle4800.prompt.i32 $F $I/code.prompt.i32 $I/code.gen.i32 $I/needle1200.prompt.i32 $I/hebrew_para.gen.i32"
} > $S/plans/long9.plan
run() { # NAME ARGS...
  local out=$TAG-long9-$1; shift
  BIN=$S/bin/$TAG/dsv41-spec bash $S/scripts/lease.sh long9 600 $CARD bash $S/scripts/spec_run.sh long9 $out \
    --layers 9 --resident --no-draft "$@" > $S/logs/$out.log 2>&1
  echo "$(date +%H:%M:%S) $out rc $? $(grep -h SPEC-DONE $S/logs/$out.log | tail -n 1)"
}
run default
run never --select-long-min 100000000
run always --select-long-min 0
python3 $S/scripts/cmp_arms.py $S/out $TAG-long9-never $TAG-long9-default $TAG-long9-always > $S/logs/$TAG-long9-cmp.txt 2>&1
echo "$(date +%H:%M:%S) long compare rc $? $(grep -h '^hashes' $S/logs/$TAG-long9-cmp.txt | tr '\n' ';')"
echo "LONG-GATE-DONE $TAG"
