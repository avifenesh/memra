#!/bin/bash
# Laptop: the memra#1311 fix after window 3 (the split MoE's gather by what a rank stores) on GB10 cb33, one task a
# lease (lease.sh), behind whatever sparklock already queued. Tags and their commits: NEW (the fix), m24 (main
# 63c351a2f4), g1b (c99798ef57, window 3's arm B).
#   builds of NEW, m24, g1b (121a); gates A of NEW (kernels; layer on the default form against fa825eee); tp of NEW at
#   four ranks on the card on peer-pull and host-canonical, then on the default transport; the counters run (tb.sh).
# usage: chain4.sh NEW NEWSHA REPO   marker CHAIN4-DONE
set -u
TAG=$1; SHA=$2; M=$3
S=/home/avifenesh/.cache/dsv41-cfix2/t2/spark
L="bash $S/lease.sh"
export UNIT=cb33
stage() { git -C $M archive --format=tar "$2" -- .cargo Cargo.toml Cargo.lock crates docs | ssh spark-cb33 "cat > .cache/dsv41-cfix2/src-$1.tar"; }
stage $TAG $SHA && stage m24 63c351a2f4 && stage g1b c99798ef57 || { echo "stage failed; CHAIN4-DONE"; exit 1; }
ssh spark-cb33 'mkdir -p ~/.cache/dsv41-cfix2/prompts' && scp -q $S/prompts/rows*.i32 spark-cb33:.cache/dsv41-cfix2/prompts/ || { echo "prompts failed; CHAIN4-DONE"; exit 1; }
for t in $TAG m24 g1b; do
  $L "build $t (121a)" out/build-$t/runs.txt BUILD-DONE 40 build.sh $t
  ssh spark-cb33 "grep -q '$t engine build rc 0' ~/.cache/dsv41-cfix2/out/build-$t/runs.txt" || { echo "$t BUILD RED; CHAIN4-DONE"; exit 1; }
done
$L "$TAG gates A (kernels, layer default form)" out/gates-$TAG/a.txt GATES-A-DONE 30 ga.sh $TAG
$L "$TAG gates B2 (tp 4 ranks peer-pull, host-canonical)" out/gates-$TAG/b2.txt GATES-B2-DONE 25 gb.sh $TAG 2 "'0,0,0,0'" "'peer-pull host-canonical'"
$L "$TAG gates B3 (tp 4 ranks default transport)" out/gates-$TAG/b3.txt GATES-B3-DONE 20 gb.sh $TAG 3 "'0,0,0,0'" "'default'"
$L "gather counters by width: $TAG, m24, g1b at 4, 2 and 3 ranks on the card, 4 layers" out/tb/runs.txt TB-DONE 30 tb.sh "'$TAG m24 g1b'" "'0,0,0,0 0,0 0,0,0'" 4
echo CHAIN4-DONE
