#!/bin/bash
# Laptop: #1286's GB10 runs at the restacked head TAG (SHA), on both Sparks, one task a lease (lease.sh):
#   cb33: stage, build (121a), then gates A (kernels; layer on the default form against fa825eee), C (decode, graph
#         door, default form), the chat tapes against the CUTLASS set
#   b23d: once cb33's binaries exist, a copy of them; then B1 (tp 2 ranks x3 transports), B2 (4 ranks peer-pull,
#         host-canonical), B3 (4 ranks default), all on the default form, and the race harness (forms 2, 3, default;
#         with the d6 debug build first, the positive control)
# usage: chain.sh TAG SHA [cb33|b23d|both (default)]   markers CHAIN-cb33-DONE, CHAIN-b23d-DONE
set -u
TAG=$1; SHA=$2; WHO=${3:-both}
S=/home/avifenesh/.cache/dsv41-memra/cutlass-decode/fix2/spark
M=/home/avifenesh/projects/memra
RC=/home/avifenesh/.cache/dsv41-memra/land1113/wt-cfix2/research/modelplan-onboarding-dsv41-20261109/perf/receipts/gates-cutlass-gb10/f4
L="bash $S/lease.sh"
stage() {  # UNIT: the reference hashes into ~/.cache/dsv41-cfix2/ref
  ssh spark-$1 'mkdir -p ~/.cache/dsv41-cfix2/ref' && scp -q $RC/save-c.sha256 spark-$1:.cache/dsv41-cfix2/ref/save-c.sha256 \
    && scp -q $RC/tapes-f4-chat.sha256 spark-$1:.cache/dsv41-cfix2/ref/tapes-ref-chat.sha256 \
    && scp -q $RC/tapes-f4-long.sha256 spark-$1:.cache/dsv41-cfix2/ref/tapes-ref-long.sha256
}
cb33() {
  export UNIT=cb33
  stage cb33 || { echo "stage cb33 failed; CHAIN-cb33-DONE"; return 1; }
  # the engine's build inputs only, as fix/mktars.sh staged them (the research tree is 8.7 GB and no engine bin reads it)
  git -C $M archive --format=tar "$SHA" -- .cargo Cargo.toml Cargo.lock crates docs | ssh spark-cb33 "cat > .cache/dsv41-cfix2/src-$TAG.tar" || return 1
  $L "build $TAG (121a)" out/build-$TAG/runs.txt BUILD-DONE 40 build.sh $TAG
  ssh spark-cb33 "grep -q '$TAG engine build rc 0' ~/.cache/dsv41-cfix2/out/build-$TAG/runs.txt" || { echo "$TAG BUILD RED; CHAIN-cb33-DONE"; return 1; }
  $L "$TAG gates A (kernels, layer default form)" out/gates-$TAG/a.txt GATES-A-DONE 30 ga.sh $TAG
  $L "$TAG gates C (decode, graph door, default form)" out/gates-$TAG/c.txt GATES-C-DONE 25 gc.sh $TAG
  $L "$TAG order-gate tapes chat" out/tapes-$TAG/chat-runs.txt TAPES-chat-DONE 22 tapes.sh $TAG chat
  echo CHAIN-cb33-DONE
}
b23d() {
  export UNIT=b23d
  stage b23d || { echo "stage b23d failed; CHAIN-b23d-DONE"; return 1; }
  waitfor --timeout 120m --every 60s --ssh spark-cb33 "grep -q 'BUILD-DONE' ~/.cache/dsv41-cfix2/out/build-$TAG/runs.txt" > /dev/null 2>&1
  ssh spark-cb33 "grep -q '$TAG engine build rc 0' ~/.cache/dsv41-cfix2/out/build-$TAG/runs.txt" || { echo "no $TAG build; CHAIN-b23d-DONE"; return 1; }
  ssh spark-b23d "mkdir -p ~/.cache/dsv41-cfix2/bins/$TAG" || return 1
  for b in dsv41-gate dsv41-bench dsv41-run dsv41-tpbench; do
    scp -q -3 spark-cb33:.cache/dsv41-cfix2/bins/$TAG/$b spark-b23d:.cache/dsv41-cfix2/bins/$TAG/$b || return 1
  done
  ssh spark-b23d "cd ~/.cache/dsv41-cfix2/bins/$TAG && chmod +x * && sha256sum *"
  $L "$TAG gates B1 (tp 2 ranks x3 transports, default form)" out/gates-$TAG/b1.txt GATES-B1-DONE 25 gb.sh $TAG 1 "'0,0'" "'peer-pull host-canonical default'"
  $L "$TAG gates B2 (tp 4 ranks peer-pull, host-canonical, default form)" out/gates-$TAG/b2.txt GATES-B2-DONE 25 gb.sh $TAG 2 "'0,0,0,0'" "'peer-pull host-canonical'"
  $L "$TAG gates B3 (tp 4 ranks default, default form)" out/gates-$TAG/b3.txt GATES-B3-DONE 20 gb.sh $TAG 3 "'0,0,0,0'" "'default'"
  $L "$TAG race harness (reps 40, forms 2, 3, default; d6 control)" out/race-$TAG/runs.txt RACE-DONE 25 race.sh $TAG 3 "'2 3 d'" 1
  echo CHAIN-b23d-DONE
}
case $WHO in
  cb33) cb33 ;;
  b23d) b23d ;;
  both) cb33 > $S/chain-cb33-$TAG.log 2>&1 &
        b23d > $S/chain-b23d-$TAG.log 2>&1 &
        wait ;;
esac
