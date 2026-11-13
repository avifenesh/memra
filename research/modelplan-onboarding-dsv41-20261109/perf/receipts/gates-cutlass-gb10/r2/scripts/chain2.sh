#!/bin/bash
# Laptop: #1286 on GB10 after the coordinator's second look, at TAG (SHA), one task a lease (lease.sh), FIFO behind
# whatever sparklock already queued (ds-sparks' landings first):
#   cb33: stage, build (121a), gates A (kernels; layer on the default form against fa825eee), the chat tapes against
#         the CUTLASS set, then race2 parts CB_PARTS
#   b23d: a copy of cb33's binaries, then race2 parts B_PARTS
# race2.sh: per part TRIPLES triples of the default form (plain, --side-delay, --poison-free) at --reps REPS, each
# triple after one run of the race's debug build d6 (the positive control; copied to cb33 from b23d first).
# usage: chain2.sh TAG SHA "CB_PARTS" "B_PARTS" [TRIPLES] [REPS]   markers CHAIN2-cb33-DONE, CHAIN2-b23d-DONE
set -u
TAG=$1; SHA=$2; CBP=$3; BP=$4; NT=${5:-2}; REPS=${6:-160}
S=/home/avifenesh/.cache/dsv41-memra/cutlass-decode/fix2/spark
M=/home/avifenesh/projects/memra
RC=/home/avifenesh/.cache/dsv41-memra/land1113/wt-cfix2/research/modelplan-onboarding-dsv41-20261109/perf/receipts/gates-cutlass-gb10/f4
L="bash $S/lease.sh"
stage() {
  ssh spark-$1 'mkdir -p ~/.cache/dsv41-cfix2/ref ~/.cache/dsv41-cfix2/bins/d6' \
    && scp -q $RC/save-c.sha256 spark-$1:.cache/dsv41-cfix2/ref/save-c.sha256 \
    && scp -q $RC/tapes-f4-chat.sha256 spark-$1:.cache/dsv41-cfix2/ref/tapes-ref-chat.sha256 \
    && scp -q $RC/tapes-f4-long.sha256 spark-$1:.cache/dsv41-cfix2/ref/tapes-ref-long.sha256
}
cb33() {
  export UNIT=cb33
  stage cb33 || { echo "stage cb33 failed; CHAIN2-cb33-DONE"; return 1; }
  ssh spark-cb33 'test -x ~/.cache/dsv41-cfix2/bins/d6/dsv41-gate' \
    || scp -q -3 spark-b23d:.cache/dsv41-cutlass/bins/d6/dsv41-gate spark-cb33:.cache/dsv41-cfix2/bins/d6/dsv41-gate
  ssh spark-cb33 'chmod +x ~/.cache/dsv41-cfix2/bins/d6/dsv41-gate'
  git -C $M archive --format=tar "$SHA" -- .cargo Cargo.toml Cargo.lock crates docs | ssh spark-cb33 "cat > .cache/dsv41-cfix2/src-$TAG.tar" || return 1
  $L "build $TAG (121a)" out/build-$TAG/runs.txt BUILD-DONE 40 build.sh $TAG
  ssh spark-cb33 "grep -q '$TAG engine build rc 0' ~/.cache/dsv41-cfix2/out/build-$TAG/runs.txt" || { echo "$TAG BUILD RED; CHAIN2-cb33-DONE"; return 1; }
  $L "$TAG gates A (kernels, layer default form)" out/gates-$TAG/a.txt GATES-A-DONE 30 ga.sh $TAG
  $L "$TAG order-gate tapes chat" out/tapes-$TAG/chat-runs.txt TAPES-chat-DONE 22 tapes.sh $TAG chat
  for p in $CBP; do
    $L "$TAG race harness part $p (default form; plain, side delay, poison free; d6 control)" out/race2-$TAG/runs-$p.txt RACE2-$p-DONE 25 race2.sh $TAG $p $NT $REPS
  done
  echo CHAIN2-cb33-DONE
}
b23d() {
  export UNIT=b23d
  stage b23d || { echo "stage b23d failed; CHAIN2-b23d-DONE"; return 1; }
  ssh spark-b23d 'test -x ~/.cache/dsv41-cfix2/bins/d6/dsv41-gate || cp -a ~/.cache/dsv41-cutlass/bins/d6/dsv41-gate ~/.cache/dsv41-cfix2/bins/d6/'
  waitfor --timeout 180m --every 60s --ssh spark-cb33 "grep -q 'BUILD-DONE' ~/.cache/dsv41-cfix2/out/build-$TAG/runs.txt" > /dev/null 2>&1
  ssh spark-cb33 "grep -q '$TAG engine build rc 0' ~/.cache/dsv41-cfix2/out/build-$TAG/runs.txt" || { echo "no $TAG build; CHAIN2-b23d-DONE"; return 1; }
  ssh spark-b23d "mkdir -p ~/.cache/dsv41-cfix2/bins/$TAG" || return 1
  for b in dsv41-gate dsv41-bench dsv41-run dsv41-tpbench; do
    scp -q -3 spark-cb33:.cache/dsv41-cfix2/bins/$TAG/$b spark-b23d:.cache/dsv41-cfix2/bins/$TAG/$b || return 1
  done
  ssh spark-b23d "cd ~/.cache/dsv41-cfix2/bins/$TAG && chmod +x * && sha256sum *"
  for p in $BP; do
    $L "$TAG race harness part $p (default form; plain, side delay, poison free; d6 control)" out/race2-$TAG/runs-$p.txt RACE2-$p-DONE 25 race2.sh $TAG $p $NT $REPS
  done
  echo CHAIN2-b23d-DONE
}
cb33 > $S/chain2-cb33-$TAG.log 2>&1 &
b23d > $S/chain2-b23d-$TAG.log 2>&1 &
wait
