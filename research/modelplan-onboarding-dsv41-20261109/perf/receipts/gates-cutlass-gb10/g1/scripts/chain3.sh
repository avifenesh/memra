#!/bin/bash
# Laptop: the memra#1311 candidate (the split MoE's gather by ranks) on GB10 cb33 at TAG (SHA), one task a lease
# (lease.sh), behind whatever sparklock already queued: build (121a), gates A (kernels; layer on the default form
# against fa825eee), tp at four ranks on the card on peer-pull and host-canonical, then on the default transport.
# usage: chain3.sh TAG SHA REPO   marker CHAIN3-DONE
set -u
TAG=$1; SHA=$2; M=$3
S=/home/avifenesh/.cache/dsv41-cfix2/t2/spark
L="bash $S/lease.sh"
export UNIT=cb33
git -C $M archive --format=tar "$SHA" -- .cargo Cargo.toml Cargo.lock crates docs | ssh spark-cb33 "cat > .cache/dsv41-cfix2/src-$TAG.tar" || { echo "stage failed; CHAIN3-DONE"; exit 1; }
$L "build $TAG (121a)" out/build-$TAG/runs.txt BUILD-DONE 40 build.sh $TAG
ssh spark-cb33 "grep -q '$TAG engine build rc 0' ~/.cache/dsv41-cfix2/out/build-$TAG/runs.txt" || { echo "$TAG BUILD RED; CHAIN3-DONE"; exit 1; }
$L "$TAG gates A (kernels, layer default form)" out/gates-$TAG/a.txt GATES-A-DONE 30 ga.sh $TAG
$L "$TAG gates B2 (tp 4 ranks peer-pull, host-canonical)" out/gates-$TAG/b2.txt GATES-B2-DONE 25 gb.sh $TAG 2 "'0,0,0,0'" "'peer-pull host-canonical'"
$L "$TAG gates B3 (tp 4 ranks default transport)" out/gates-$TAG/b3.txt GATES-B3-DONE 20 gb.sh $TAG 3 "'0,0,0,0'" "'default'"
echo CHAIN3-DONE
