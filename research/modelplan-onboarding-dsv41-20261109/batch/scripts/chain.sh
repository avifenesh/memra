#!/bin/bash
# The batch lane's gate chain on one card, back to back short leases on the binaries kept under spec/bin/TAG
# (memra-server, dsv41-run, dsv41-spec), all on the 9-layer resident trunk:
#   1. keyed: plan keyed9 twice (the copy path with --copy-keys, then the default keyed path) and the byte compare
#      of every verify-rows / batch-rows job's hashes.txt between the two
#   2. wide: plan wide9 with --batch-max 16 (16 sequences: the byte gate of wide calls, one call of 16 against two
#      of 8), and plan wide20 with --batch-max 20 (the one-rank reference of the two-rank wide case)
#   3. nrank: two ranks on the one card (--devices 0,0), on peer-pull and on device-push: plan keyed9 and its hash
#      compare against the one-rank keyed arm, plan wide20 (a gather past device-push's bound), the served chat gate
#   3b. pair (only when STEPS names it): the same on two cards, PAIR=01 style, through the 4-card lane's wrapper
#   4. gate: plan gate9b (verify rows, batched rows, rows-time, batch-time with the launch tally, the draft tapes)
#   5. plans: extra plans named in PLANS (space separated), each as its own lease
#   6. serve: the served chat gate on one rank
#   7. ab: the served A/B (MEMRA_DSV41_BATCH_ROWS 1 vs 8, A B B A A B) through the shared client
# usage: chain.sh TAG CARD            env: PLANS="draft9 ..."  STEPS="keyed wide nrank gate plans serve ab" (default; add pair with PAIR=01)
set -u
TAG=$1; CARD=$2
S=$LANE/spec
export TBIN=$S/bin/$TAG
STEPS=${STEPS:-keyed gate wide nrank plans serve ab}
for b in memra-server dsv41-run dsv41-spec; do
  [ -x $TBIN/$b ] || { echo "missing $TBIN/$b"; exit 2; }
done
has() { case " $STEPS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
spec() { # NAME PLAN OUT ARGS...
  local name=$1 plan=$2 out=$3; shift 3
  BIN=$TBIN/dsv41-spec bash $S/scripts/lease.sh $name 600 $CARD bash $S/scripts/spec_run.sh $plan $out "$@" \
    > $S/logs/$out.log 2>&1
  echo "$(date +%H:%M:%S) $out rc $? $(grep -h SPEC-DONE $S/logs/$out.log | tail -n 1)"
}
if has keyed; then
  spec k9c keyed9 $TAG-keyed9-copy --layers 9 --resident --no-draft --copy-keys
  spec k9k keyed9 $TAG-keyed9-on --layers 9 --resident --no-draft
  python3 $S/scripts/cmp_arms.py $S/out $TAG-keyed9-copy $TAG-keyed9-on > $S/logs/$TAG-keyed9-cmp.txt 2>&1
  echo "$(date +%H:%M:%S) keyed compare rc $? $(grep -h '^hashes' $S/logs/$TAG-keyed9-cmp.txt)"
fi
if has wide; then
  # calls of up to 16 rows (the GEMMs' general bodies): the byte gate and one wide call against two calls of 8
  spec w16 wide9 $TAG-wide9 --layers 9 --resident --no-draft --batch-max 16
  python3 $S/scripts/cmp_arms.py $S/out $TAG-wide9 > $S/logs/$TAG-wide9-cmp.txt 2>&1
  # one call of 20 rows on one rank: the reference of the two-rank step's wide case
  spec w20 wide20 $TAG-wide20 --layers 9 --resident --no-draft --batch-max 20
fi
if has nrank; then
  # The rank group, two ranks on the one card (--devices 0,0), on each device transport. Experts on the card and
  # more than one rank: the MoE is the split one. Per transport:
  #  * plan keyed9: its batch-rows and verify-rows jobs pass when a batched (or verification) row on two ranks
  #    equals the same sequence's own call on two ranks; the hash compare against the one-rank keyed arm then says
  #    that call's bytes are the one-rank call's (needs the keyed step's output). Its rows-time and batch-time
  #    (distinct8, 1/2/4/8 sequences, launch table) jobs are the first two-rank curves.
  #  * plan wide20 with --batch-max 20: a call of 20 rows, whose expert hidden gather (7 x rows x 2304 bytes a
  #    rank at two ranks) is past device-push's 256 KiB a rank, so that gather takes the copy engine between
  #    pushed ones; compared with the one-rank wide20 run of the wide step when it exists.
  #  * the served chat gate with two ranks: five at once == solo == the one-rank native references.
  for tr in peer-pull device-push; do
    t=${tr%%-*}
    export MEMRA_TP_TRANSPORT=$tr
    spec k9n2$t keyed9 $TAG-keyed9-n2-$t --layers 9 --resident --no-draft --devices 0,0
    python3 $S/scripts/cmp_arms.py $S/out $TAG-keyed9-on $TAG-keyed9-n2-$t > $S/logs/$TAG-keyed9-n2-$t-cmp.txt 2>&1
    echo "$(date +%H:%M:%S) two-rank $tr compare rc $? $(grep -h '^hashes' $S/logs/$TAG-keyed9-n2-$t-cmp.txt)"
    spec w20n2$t wide20 $TAG-wide20-n2-$t --layers 9 --resident --no-draft --batch-max 20 --devices 0,0
    if [ -f $S/out/$TAG-wide20/stdout.jsonl ]; then
      python3 $S/scripts/cmp_arms.py $S/out $TAG-wide20 $TAG-wide20-n2-$t > $S/logs/$TAG-wide20-n2-$t-cmp.txt 2>&1
      echo "$(date +%H:%M:%S) two-rank $tr wide compare rc $? $(grep -h '^hashes' $S/logs/$TAG-wide20-n2-$t-cmp.txt)"
    fi
    bash $S/scripts/lease.sh gate9n2$t 600 $CARD bash $S/scripts/serve_gate9.sh gate9-$TAG-n2-$t \
      MEMRA_DSV41_DEVICES=0,0 MEMRA_TP_TRANSPORT=$tr > $S/logs/gate9-$TAG-n2-$t.log 2>&1
    echo "$(date +%H:%M:%S) two-rank $tr served gate rc $? $(grep -c PASS $S/logs/gate9-$TAG-n2-$t.log) pass lines"
    unset MEMRA_TP_TRANSPORT
  done
fi
if has pair; then
  # the rank group on two cards (PAIR names them, e.g. 01), through the 4-card lane's lease wrapper, which holds
  # every card's box lock for the step: the same plans, compares and served gate as the one-card two-rank step
  P=${PAIR:?PAIR names the two cards, e.g. 01}
  TL=$LANE/bin/tp4_lease.sh
  for tr in peer-pull device-push; do
    t=${tr%%-*}
    export MEMRA_TP_TRANSPORT=$tr
    for job in "keyed9 keyed9-p2-$t" "wide20 wide20-p2-$t --batch-max 20"; do
      set -- $job; plan=$1; out=$TAG-$2; shift 2
      BIN=$TBIN/dsv41-spec bash $TL spec-$plan-p2$t 600 $P bash $S/scripts/spec_run.sh $plan $out --layers 9 \
        --resident --no-draft --devices 0,1 "$@" > $S/logs/$out.log 2>&1
      echo "$(date +%H:%M:%S) $out rc $? $(grep -h SPEC-DONE $S/logs/$out.log | tail -n 1)"
    done
    python3 $S/scripts/cmp_arms.py $S/out $TAG-keyed9-on $TAG-keyed9-p2-$t > $S/logs/$TAG-keyed9-p2-$t-cmp.txt 2>&1
    echo "$(date +%H:%M:%S) two-card $tr compare rc $? $(grep -h '^hashes' $S/logs/$TAG-keyed9-p2-$t-cmp.txt)"
    if [ -f $S/out/$TAG-wide20/stdout.jsonl ]; then
      python3 $S/scripts/cmp_arms.py $S/out $TAG-wide20 $TAG-wide20-p2-$t > $S/logs/$TAG-wide20-p2-$t-cmp.txt 2>&1
      echo "$(date +%H:%M:%S) two-card $tr wide compare rc $? $(grep -h '^hashes' $S/logs/$TAG-wide20-p2-$t-cmp.txt)"
    fi
    bash $TL spec-gate9p2$t 600 $P bash $S/scripts/serve_gate9.sh gate9-$TAG-p2-$t MEMRA_DSV41_DEVICES=0,1 \
      MEMRA_TP_TRANSPORT=$tr > $S/logs/gate9-$TAG-p2-$t.log 2>&1
    echo "$(date +%H:%M:%S) two-card $tr served gate rc $? $(grep -c PASS $S/logs/gate9-$TAG-p2-$t.log) pass lines"
    unset MEMRA_TP_TRANSPORT
  done
fi
if has gate; then
  spec g9b gate9b $TAG-gate9b --layers 9 --resident
  python3 $S/scripts/cmp_arms.py $S/out $TAG-gate9b > $S/logs/$TAG-gate9b-cmp.txt 2>&1
fi
if has plans; then
  for p in ${PLANS:-}; do
    spec $p $p $TAG-$p --layers 9 --resident
  done
fi
if has serve; then
  bash $S/scripts/lease.sh gate9 600 $CARD bash $S/scripts/serve_gate9.sh gate9-$TAG > $S/logs/gate9-$TAG.log 2>&1
  echo "$(date +%H:%M:%S) served gate rc $? $(grep -c PASS $S/logs/gate9-$TAG.log) pass lines"
fi
if has ab; then
  i=0
  for arm in A B B A A B; do
    i=$((i + 1))
    n=8; [ $arm = A ] && n=1
    EXTRA_BENCH=--no-smoke-gate bash $S/scripts/lease.sh ab-$TAG-r$i 900 $CARD bash $S/scripts/serve_bench.sh \
      ab-$TAG/r$i-$arm decode_c1,conc8 3 MEMRA_DSV41_TRUNK_LAYERS=9 MEMRA_CTX=8192 MEMRA_DSV41_BATCH_ROWS=$n \
      > $S/logs/ab-$TAG-r$i-$arm.log 2>&1
    echo "$(date +%H:%M:%S) run $i arm $arm rc $?"
  done
  python3 $S/scripts/ab_summary.py $S/out/ab-$TAG
fi
echo "CHAIN-DONE $TAG"
