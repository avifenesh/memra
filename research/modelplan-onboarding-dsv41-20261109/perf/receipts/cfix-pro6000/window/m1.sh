#!/bin/bash
# wrx, inside a one-card lease (win.sh m1): #1286 item 3 on the NativeQualified topology (one RTX PRO 6000, the routed
# experts in pinned host memory), at the PR head:
#   1. the residency byte check. HostExperts::fill repacks each expert's scale codes into CUTLASS's layout as it
#      stores them, so the residency route is compared with the streamed one byte for byte: dsv41-run teacher-forced on
#      the 13 chat tapes (517 calls, one pass), three runs of one binary:
#        stream  the streamed one-card runner (every layer loaded, used, freed: the loader's repack)
#        host    --experts-host, the serving residency, the cache as large as the card allows
#        floor   --experts-host --cache-slots 384 (one layer's experts: every layer evicts the last, so every expert
#                of every layer goes through fill and a slot copy)
#      every file of every case (top_ids, top_vals, lse, argmax, the full logits of 8 steps) compared with cmp;
#      meta.json is left out (it carries timings)
#   2. memra model verify checkpoint, native side dsv41-run --experts-host through the tree's wrapper (CheckpointParity)
#   3. memra model verify serve, --native-runner memra-server (one card: the host residency)
# A step starts only with 400 GB of host memory available (the expert store pins 289 GB).
# usage: m1.sh BINS_TAG TREE   (TREE: the head's tree, /home/access/dsv41/src-m1-<sha10>, put there by cfix2_bins.sh)
#        -> /home/access/dsv41/cfix2/m1-<sha10>/{run.txt,res-*,verify/}; marker M1-DONE
set -u
D=/home/access/dsv41
B=$D/perf-bins/${1:?bins tag}
SRC=$(readlink -f ${2:?tree})
H=${SRC##*/src-m1-}
C=DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
S=$SRC/research/modelplan-onboarding-dsv41-20261109/serve
RCP=$SRC/research/modelplan-onboarding-dsv41-20261109/gpu/receipts/checkpoint/verify-capital-rel
O=$D/cfix2/m1-$H
OUT=$O/verify
mkdir -p $OUT
cd $D || exit 1
avail() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo; }
say() { echo "$(date +%T) $*" | tee -a $O/run.txt; }
head=$(cut -d' ' -f1 $B/SOURCE)
case "$head" in "$H"*) ;; *) say "binaries ($head) and tree ($H) differ: stop"; exit 2 ;; esac
(cd $B && sha256sum dsv41-run memra memra-server) > $O/bins.sha256
say "head $head, cards: ${CUDA_VISIBLE_DEVICES:-?}, $(avail) GB available"
L=$O/chat-forced.txt
cp $D/dec/chat-forced.txt $L
for arm in stream host floor; do
  case $arm in stream) X="" ;; host) X="--experts-host" ;; floor) X="--experts-host --cache-slots 384" ;; esac
  if [ "$(avail)" -lt 400 ]; then say "$arm not started: $(avail) GB available"; continue; fi
  rm -rf $O/res-$arm
  t0=$(date +%s)
  env MEMRA_FULL_PREC=1 $B/dsv41-run $C --token-map $TM --cases $L --out $O/res-$arm $X > $O/res-$arm.log 2>&1
  say "$arm rc $? $(( $(date +%s) - t0 )) s; $(grep -h 'dsv41-run: [0-9]* forwards' $O/res-$arm.log | tail -n 1); $(grep -ahi 'misses\|evictions' $O/res-$arm.log | tail -n 1)"
done
(cd $O/res-stream && find . -type f ! -name meta.json | sort) > $O/files.txt
for arm in host floor; do
  n=0; same=0; bad=""
  while read -r f; do
    n=$((n + 1))
    if cmp -s "$O/res-stream/$f" "$O/res-$arm/$f"; then same=$((same + 1)); else bad="$bad $f"; fi
  done < $O/files.txt
  say "residency $arm against stream: $same / $n files byte-identical${bad:+; differ:$bad}"
done
(cd $O && for a in stream host floor; do (cd res-$a 2>/dev/null && find . -type f ! -name meta.json | sort | xargs sha256sum | sed "s|^|$a |"); done) > $O/res.sha256
if [ "$(avail)" -lt 400 ]; then say "verify not started: $(avail) GB available"; say "M1-DONE"; exit 3; fi
gunzip -c $RCP/vendor-cuda-capital.tsv.gz > $O/vendor-cuda-capital.tsv
env DSV41_RUN=$B/dsv41-run MEMRA_DSV41_TOKEN_MAP=$TM $B/memra model verify checkpoint $C \
  --against deepseek_v41 --oracle $O/vendor-cuda-capital.tsv --native-runner $S/scripts/dsv41-run-experts-host \
  --out $OUT > $OUT/verify-checkpoint-res.log 2>&1
say "checkpoint rc $? $(grep -o 'verified=[A-Za-z]*' $OUT/verify-checkpoint-res.log | tail -1)"
env MEMRA_DSV41_TOKEN_MAP=$TM MEMRA_SERVE_VERIFY_TIMEOUT_S=1500 $B/memra model verify serve $C \
  --against deepseek_v41 --out $OUT --native-runner $B/memra-server > $OUT/verify-serve.log 2>&1
say "serve rc $? $(grep -o 'verified=[A-Za-z]*' $OUT/verify-serve.log | tail -1)"
cat $OUT/gates.txt 2>/dev/null | tee -a $O/run.txt
say "M1-DONE"
