#!/bin/bash
# Laptop: the review fold of memra#1327 (the gather's arithmetic through one helper; 9 and 10 rows in the tp gate) on
# GB10 cb33 at TAG (SHA), one task a lease (lease.sh), normal tickets behind whatever sparklock already queued:
# build (121a), the kernel gate, tp at four ranks on the card on peer-pull and host-canonical, then on the default
# transport (the gate now runs 9 and 10 rows beside 12, 16 and 32).
# usage: chain6.sh TAG SHA REPO   marker CHAIN6-DONE
set -u
TAG=$1; SHA=$2; M=$3
S=/home/avifenesh/.cache/dsv41-cfix2/t2/spark
L="bash $S/lease.sh"
export UNIT=cb33
git -C $M archive --format=tar "$SHA" -- .cargo Cargo.toml Cargo.lock crates docs | ssh spark-cb33 "cat > .cache/dsv41-cfix2/src-$TAG.tar" || { echo "stage failed; CHAIN6-DONE"; exit 1; }
$L "build $TAG (121a)" out/build-$TAG/runs.txt BUILD-DONE 40 build.sh $TAG
ssh spark-cb33 "grep -q '$TAG engine build rc 0' ~/.cache/dsv41-cfix2/out/build-$TAG/runs.txt" || { echo "$TAG BUILD RED; CHAIN6-DONE"; exit 1; }
$L "$TAG kernel gate" out/gates-$TAG/k.txt GATES-K-DONE 20 gk.sh $TAG
$L "$TAG gates B2 (tp 4 ranks peer-pull, host-canonical; 9 and 10 rows added)" out/gates-$TAG/b2.txt GATES-B2-DONE 25 gb.sh $TAG 2 "'0,0,0,0'" "'peer-pull host-canonical'"
$L "$TAG gates B3 (tp 4 ranks default transport; 9 and 10 rows added)" out/gates-$TAG/b3.txt GATES-B3-DONE 20 gb.sh $TAG 3 "'0,0,0,0'" "'default'"
echo CHAIN6-DONE
