#!/usr/bin/env bash
# GB10 box, batch lane: build TAG from ~/hy4flash/batch/src, the batch identity gate
# (hy4f-spec batch: alone vs batched, bit for bit), check3 (c=1 vs the PR 8 tapes), then the
# served readings of TAG (drafts off, gate) and base again (drafts off, gate): A B A over the
# baseline chain's A. usage: chain-new.sh TAG [skip-base]
set -u
tag=$1
D=$HOME/hy4flash/batch
L=$HOME/hy4flash/logs
cd $D
echo "CHAIN-NEW START $(date '+%F %T') $(cat src/HEAD.txt)"
bash ./build.sh $tag > $L/batch-build-$tag.log 2>&1
grep -q BUILD_EXIT=0 $L/batch-build-$tag.log || { echo BUILD-FAILED; tail -n 30 $L/batch-build-$tag.log; exit 1; }
B=$D/bins-$tag
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -- \
  $B/hy4f-spec batch $HOME/hy4flash/hf-stage $HOME/hy4flash/pr7/requests/long.json 48 --parts 4 \
  > $L/batch-ident-$tag.log 2>&1
brc=$?
echo "batch-ident rc=$brc $(tail -n 1 $L/batch-ident-$tag.log)"
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
bash $HOME/hy4flash/perf/check3.sh batch-$tag $B/hy4f-gpu; tail -n 1 $L/perf-check3-batch-$tag.log
if [ $brc -ne 0 ] || ! grep -q 'CHECK3_DONE 3/3' $L/perf-check3-batch-$tag.log; then echo "GATES-FAILED"; exit 1; fi
bash ./gate.sh $tag-chatgate $B/memra-server; echo "chat-gate rc=$? $(tail -n 1 $D/serve/$tag-chatgate/chat-gate.log)"
bash ./verify.sh $tag; echo "verify rc=$?"
[ "${2:-}" = "gates-only" ] && { echo "CHAIN-NEW-DONE $(date '+%F %T')"; exit 0; }
bash ./serve.sh $tag-off $B/memra-server MEMRA_HY4F_DRAFT=0; echo "$tag-off rc=$? $(date '+%T')"
bash ./serve.sh $tag-gate $B/memra-server; echo "$tag-gate rc=$? $(date '+%T')"
if [ "${2:-}" != "skip-base" ]; then
  bash ./serve.sh base-$tag-off $D/bins-base/memra-server MEMRA_HY4F_DRAFT=0; echo "base-$tag-off rc=$? $(date '+%T')"
  bash ./serve.sh base-$tag-gate $D/bins-base/memra-server; echo "base-$tag-gate rc=$? $(date '+%T')"
fi
bash ./prof-serve.sh $tag-off-c4 $B/memra-server 4 10 MEMRA_HY4F_DRAFT=0; echo "prof off rc=$? $(date '+%T')"
bash ./prof-serve.sh $tag-gate-c4 $B/memra-server 4 10; echo "prof gate rc=$? $(date '+%T')"
[ -f $D/prof/base-off-c4.nsys-rep ] || { bash ./prof-serve.sh base-off-c4 $D/bins-base/memra-server 4 10 MEMRA_HY4F_DRAFT=0; echo "prof base off rc=$? $(date '+%T')"; }
echo "CHAIN-NEW-DONE $(date '+%F %T')"
