#!/bin/bash
# Inside a one-card lease: dsv41-spec over a plan, in the serving route's one-card residency.
# usage: spec_run.sh PLAN_NAME OUT_NAME [dsv41-spec args...]      BIN=<binary> runs a kept build
set -uo pipefail
PLAN=$1; OUT=$2; shift 2
D=$LANE
S=$D/spec
mkdir -p $S/out/$OUT $S/logs
echo "host memory before: $(free -g | sed -n 2p)"
MEMRA_FULL_PREC=1 MEMRA_DSV41_TOKEN_MAP=$D/oracle/engram_constants.token_map.i32 \
  ${BIN:-$D/target-spec/release/dsv41-spec} $D/DeepSeek-V4.1-Flash-1017 --plan $S/plans/$PLAN.plan --out $S/out/$OUT "$@" \
  > $S/out/$OUT/stdout.jsonl 2> $S/out/$OUT/stderr.log
rc=$?
tail -n 4 $S/out/$OUT/stderr.log | cut -c1-400
cat $S/out/$OUT/stdout.jsonl | cut -c1-1500
echo "spec_run rc $rc"
exit $rc
