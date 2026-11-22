#!/usr/bin/env bash
# GB10 box, batch lane, the final head TAG: build, the batch identity gates (4 parts, every other
# drafting; 8 parts, all drafting), check3, the served-answer gates (drafts on), verify checkpoint
# + serve, then the readings in the order base-off, TAG-off, TAG-gate, base-gate (A B B A), the
# step prices (hy4f-spec cost: base, TAG, base), and nsys of a c=4 decode (off, gate).
# usage: chain-final.sh TAG
set -u
tag=$1
D=$HOME/hy4flash/batch
L=$HOME/hy4flash/logs
cd $D
echo "CHAIN-FINAL START $(date '+%F %T') $(cat src/HEAD.txt)"
bash ./build.sh $tag > $L/batch-build-$tag.log 2>&1
grep -q BUILD_EXIT=0 $L/batch-build-$tag.log || { echo BUILD-FAILED; tail -n 30 $L/batch-build-$tag.log; exit 1; }
B=$D/bins-$tag
ident() {
  waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
  systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -- \
    $B/hy4f-spec batch $HOME/hy4flash/hf-stage $HOME/hy4flash/pr7/requests/long.json 48 "$@"
}
ident --parts 4 > $L/batch-ident-$tag.log 2>&1; r1=$?
echo "batch-ident 4 rc=$r1 $(tail -n 1 $L/batch-ident-$tag.log)"
ident --parts 8 --all-draft > $L/batch-ident8-$tag.log 2>&1; r2=$?
echo "batch-ident 8 rc=$r2 $(tail -n 1 $L/batch-ident8-$tag.log)"
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
bash $HOME/hy4flash/perf/check3.sh batch-$tag $B/hy4f-gpu; tail -n 1 $L/perf-check3-batch-$tag.log
if [ $r1 -ne 0 ] || [ $r2 -ne 0 ] || ! grep -q 'CHECK3_DONE 3/3' $L/perf-check3-batch-$tag.log; then echo "GATES-FAILED"; exit 1; fi
bash ./gate.sh $tag-chatgate $B/memra-server; echo "chat-gate rc=$? $(tail -n 1 $D/serve/$tag-chatgate/chat-gate.log)"
bash ./verify.sh $tag; echo "verify rc=$?"
bash ./serve.sh base3-off $D/bins-base/memra-server MEMRA_HY4F_DRAFT=0; echo "base3-off rc=$? $(date '+%T')"
bash ./serve.sh $tag-off $B/memra-server MEMRA_HY4F_DRAFT=0; echo "$tag-off rc=$? $(date '+%T')"
bash ./serve.sh $tag-gate $B/memra-server; echo "$tag-gate rc=$? $(date '+%T')"
bash ./serve.sh base3-gate $D/bins-base/memra-server; echo "base3-gate rc=$? $(date '+%T')"
cost() {
  waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
  systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -- \
    $1/hy4f-spec cost $HOME/hy4flash/hf-stage $HOME/hy4flash/pr7/requests/long.json --reps 8 --temperature 0.9
}
cost $D/bins-base > $L/batch-cost-base-a.log 2>&1; echo "cost base rc=$?"
cost $B > $L/batch-cost-$tag.log 2>&1; echo "cost $tag rc=$?"
cost $D/bins-base > $L/batch-cost-base-b.log 2>&1; echo "cost base rc=$?"
bash ./prof-serve.sh $tag-off-c4 $B/memra-server 4 10 MEMRA_HY4F_DRAFT=0; echo "prof off rc=$? $(date '+%T')"
bash ./prof-serve.sh $tag-gate-c4 $B/memra-server 4 10; echo "prof gate rc=$? $(date '+%T')"
echo "CHAIN-FINAL-DONE $(date '+%F %T')"
