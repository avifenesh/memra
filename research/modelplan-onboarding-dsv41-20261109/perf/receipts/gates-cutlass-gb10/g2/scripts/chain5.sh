#!/bin/bash
# Laptop: the memra#1311 fix after window 3, rebased onto main 0db086f87a, on GB10 cb33, one task a lease (lease.sh),
# behind whatever sparklock already queued. Tags and their commits: NEW (the fix's head), m24 (main 63c351a2f4, window
# 3's main arm), g1b (c99798ef57, window 3's arm B). A tag whose binaries are already on the unit is not rebuilt.
#   builds (121a); gates A of NEW (kernels; layer on the default form against fa825eee); tp of NEW at four ranks on
#   the card on peer-pull and host-canonical, then on the default transport; the counters run (tb.sh).
# usage: chain5.sh NEW NEWSHA REPO   marker CHAIN5-DONE
set -u
TAG=$1; SHA=$2; M=$3
S=/home/avifenesh/.cache/dsv41-cfix2/t2/spark
L="bash $S/lease.sh"
export UNIT=cb33
stage() { git -C $M archive --format=tar "$2" -- .cargo Cargo.toml Cargo.lock crates docs | ssh spark-cb33 "cat > .cache/dsv41-cfix2/src-$1.tar"; }
have() { ssh spark-cb33 "grep -q '$1 engine build rc 0' ~/.cache/dsv41-cfix2/out/build-$1/runs.txt 2>/dev/null && test -x ~/.cache/dsv41-cfix2/bins/$1/dsv41-tpbench"; }
ssh spark-cb33 'mkdir -p ~/.cache/dsv41-cfix2/prompts' && scp -q $S/prompts/rows*.i32 spark-cb33:.cache/dsv41-cfix2/prompts/ || { echo "prompts failed; CHAIN5-DONE"; exit 1; }
for pair in "$TAG $SHA" "m24 63c351a2f4" "g1b c99798ef57"; do
  set -- $pair
  if have $1; then echo "$1: binaries already built"; continue; fi
  stage $1 $2 || { echo "stage $1 failed; CHAIN5-DONE"; exit 1; }
  $L "build $1 (121a)" out/build-$1/runs.txt BUILD-DONE 40 build.sh $1
  have $1 || { echo "$1 BUILD RED; CHAIN5-DONE"; exit 1; }
done
$L "$TAG gates A (kernels, layer default form)" out/gates-$TAG/a.txt GATES-A-DONE 30 ga.sh $TAG
$L "$TAG gates B2 (tp 4 ranks peer-pull, host-canonical)" out/gates-$TAG/b2.txt GATES-B2-DONE 25 gb.sh $TAG 2 "'0,0,0,0'" "'peer-pull host-canonical'"
$L "$TAG gates B3 (tp 4 ranks default transport)" out/gates-$TAG/b3.txt GATES-B3-DONE 20 gb.sh $TAG 3 "'0,0,0,0'" "'default'"
$L "gather counters by width: $TAG, m24, g1b at 4, 2 and 3 ranks on the card, 4 layers" out/tb/runs.txt TB-DONE 30 tb.sh "'$TAG m24 g1b'" "'0,0,0,0 0,0 0,0,0'" 4
echo CHAIN5-DONE
