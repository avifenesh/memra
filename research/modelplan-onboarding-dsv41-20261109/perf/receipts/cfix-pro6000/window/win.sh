#!/bin/bash
# wrx, a DSv4.1 window for #1286 (the CUTLASS lane): the review's PRO 6000 items, each step its own lease through
# bin/perf_runc.sh / bin/perf_cfix_four.sh / bin/perf_prefill_ab.sh (tp4_lease.sh: QUIET, PAUSE, every card's box
# lock, 250 ms telemetry, slowdown counters).
#   four      item 1 (+6): four cards, 40 layers, A=$A against B=$B, A B B A B A A B (4 runs an arm), one-row decode and
#             blocks of 4/8/9/12/16 rows, B's kernel gate and tp gates (default, peer-pull, host-canonical) first
#   tok4      item 6: the decode token tape of B against the CUTLASS tree before the fix (perf-bins/cutlass,
#             498473abd8), four cards, 40 layers, A B, 8-row block
#   prefill4  item 2: prefill A/B on four cards, 40 layers, 4,096 and 32,768 tokens, 3 rounds, A B B A twice
#   two       item 6: one more pair block, cards $PAIR, 20 layers, A B B A B A A B, tp gates first (as cfix-two ran)
#   prefill2  item 2: prefill A/B on the pair, 20 layers, 4,096 / 32,768 / 131,072 tokens, 3 rounds, A B B A twice
#   stress2   item 4: the race harness at one rank per card on the pair, default dense form (stress2.sh)
#   m1        item 3: card $M1CARD, the residency byte check, CheckpointParity and verify serve (m1.sh)
# usage (from /home/access/dsv41): (nohup setsid bash cfix2/win.sh "four tok4 prefill4 two prefill2 stress2 m1") \
#          > cfix2/win.log 2>&1 < /dev/null &
# env: A (default route2: the PR's base), B (default cfix2: the PR head), PAIR (default 23), M1CARD (default 2),
#      M1_TREE (m1: the head's tree, default src-m1-<B's SOURCE sha10>). Marker CFIX2-WIN-DONE; summary cfix2/win-summary.txt
set -u
D=/home/access/dsv41
P=$D/perf-bins
A=${A:-route2}; B=${B:-cfix2}; PAIR=${PAIR:-23}; M1CARD=${M1CARD:-2}
S=$D/cfix2
RP=$D/receipts-perf
SUM=$S/win-summary.txt
cd $D || exit 1
say() { echo "$(date +%T) $*" | tee -a $SUM; }
for t in $A $B; do
  [ -x $P/$t/dsv41-tpbench ] && [ -x $P/$t/dsv41-gate ] || { say "missing $P/$t"; exit 2; }
  say "$t: $(cat $P/$t/SOURCE)"
done
for ph in ${1:-four tok4 prefill4 two prefill2 stress2 m1}; do
  say "phase $ph start"
  case $ph in
  four)
    env K3_ARM_A=$A K3_ARM_B=$B K3_TAG=cfix2-four K3_CARDS=0123 K3_DEV=0,1,2,3 K3_LAYERS=40 \
      K3_ORDER="A B B A B A A B" K3_KGATE=1 K3_GATE_T="default peer-pull host-canonical" \
      bash bin/perf_cfix_four.sh 780 > $S/four.out 2>&1
    say "four rc $?"
    python3 $S/per_run.py $RP/cfix2-four > $RP/cfix2-four/per_run.md 2>&1
    grep -E '^(kgate|gate-)|SKIP' $RP/cfix2-four/runs.txt >> $SUM
    sed -n '/^Arms/,/^Telemetry/p' $RP/cfix2-four/per_run.md >> $SUM
    ;;
  tok4)
    env K3_ARM_A=cutlass K3_ARM_B=$B K3_TAG=cfix2-tok4 K3_CARDS=0123 K3_DEV=0,1,2,3 K3_LAYERS=40 \
      K3_ORDER="A B" K3_ROWS="8" K3_GATES=0 bash bin/perf_cfix_four.sh 240 > $S/tok4.out 2>&1
    say "tok4 rc $?"
    python3 $S/per_run.py $RP/cfix2-tok4 > $RP/cfix2-tok4/per_run.md 2>&1
    sed -n '/^Decode token/,/^Telemetry/p' $RP/cfix2-tok4/per_run.md >> $SUM
    ;;
  prefill4)
    env CARDS=0123 PRE_REPEAT=2 PRE_TRANSPORT=default ROUNDS=3 LEASE_TO=1800 \
      PROMPTS=$D/bench/prompts/prefill-4096.i32,$D/bench/prompts/prefill-32768.i32 \
      bash bin/perf_prefill_ab.sh cfix2-four 0,1,2,3 40 $P/$A $P/$B > $S/prefill4.out 2>&1
    say "prefill4 rc $?"
    tail -n 12 $RP/prefillab-cfix2-four/summary.md >> $SUM
    ;;
  two)
    env K3_ARM_A=$A K3_ARM_B=$B K3_TAG=cfix2-two K3_CARDS=$PAIR K3_DEV=0,1 K3_LAYERS=20 \
      K3_ORDER="A B B A B A A B" K3_GATE_T="default peer-pull host-canonical" \
      bash bin/perf_cfix_four.sh 600 > $S/two.out 2>&1
    say "two rc $?"
    python3 $S/per_run.py $RP/cfix2-two > $RP/cfix2-two/per_run.md 2>&1
    grep -E '^gate-|SKIP' $RP/cfix2-two/runs.txt >> $SUM
    sed -n '/^Arms/,/^Telemetry/p' $RP/cfix2-two/per_run.md >> $SUM
    ;;
  prefill2)
    env CARDS=$PAIR PRE_REPEAT=2 PRE_TRANSPORT=default ROUNDS=3 LEASE_TO=1800 \
      PROMPTS=$D/bench/prompts/prefill-4096.i32,$D/bench/prompts/prefill-32768.i32,$D/bench/prompts/prefill-131072.i32 \
      bash bin/perf_prefill_ab.sh cfix2-pair 0,1 20 $P/$A $P/$B > $S/prefill2.out 2>&1
    say "prefill2 rc $?"
    tail -n 12 $RP/prefillab-cfix2-pair/summary.md >> $SUM
    ;;
  stress2)
    bash bin/perf_runc.sh cfix2/stress2 1500 $PAIR bash $S/stress2.sh $B > $S/stress2.out 2>&1
    say "stress2 rc $?"
    cat $S/stress2/runs.txt >> $SUM 2>/dev/null
    ;;
  m1)
    sha=$(cut -c1-10 $P/$B/SOURCE)
    # the expert store pins 289 GB of host memory: no lease is taken unless 400 GB are available now
    av=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    if [ "$av" -lt 400 ]; then
      say "m1 SKIPPED: $av GB of host memory available, 400 needed (the residency pins 289 GB); rerun the phase on a quiet box"
      continue
    fi
    [ -d "${M1_TREE:-$D/src-m1-$sha}/research" ] || { say "m1 SKIPPED: no tree at ${M1_TREE:-$D/src-m1-$sha}"; continue; }
    [ -x $P/$B/memra ] && [ -x $P/$B/memra-server ] && [ -x $P/$B/dsv41-run ] || { say "m1 SKIPPED: $P/$B lacks memra, memra-server or dsv41-run"; continue; }
    say "m1: $av GB available"
    bash bin/perf_runc.sh cfix2/m1 3000 $M1CARD bash $S/m1.sh $B ${M1_TREE:-$D/src-m1-$sha} > $S/m1.out 2>&1
    say "m1 rc $?"
    cat $S/m1-$sha/run.txt >> $SUM 2>/dev/null
    ;;
  *) say "unknown phase $ph" ;;
  esac
done
say CFIX2-WIN-DONE
