#!/bin/bash
# wrx, inside a one-card lease (win_land.sh phase 3): #1281 M1, the NativeQualified record deepseek_v41-gpu-1x-pro6000
# re-verified at the final landing head, the way the 87bbe3bfc8 bundle was made (serve/receipts/verify-res):
#   1. memra model verify checkpoint, native side on the residency (dsv41-run --experts-host via the tree's wrapper)
#   2. memra model verify serve, --native-runner memra-server
# Binaries: perf-bins/$M1_TAG (dev desk build of the head); tree: src-m1 (bins_to_wrx.sh TREE=1, the same commit).
# The checkpoint dir is relative: runs with cwd /home/access/dsv41. No MEMRA_FULL_PREC on the serve steps.
# Out: m1/verify-<head10>/ (the bundle) + m1/*.log; marker M1-DONE.
set -u
D=/home/access/dsv41
B=$D/perf-bins/${M1_TAG:-land-final}
SRC=$(readlink -f $D/src-m1)
H=${SRC##*/src-m1-}
TM=$D/oracle/engram_constants.token_map.i32
S=$SRC/research/modelplan-onboarding-dsv41-20261109/serve
RCP=$SRC/research/modelplan-onboarding-dsv41-20261109/gpu/receipts/checkpoint/verify-capital-rel
M=$D/m1; OUT=$M/verify-$H
mkdir -p $OUT
cd $D || exit 1
head=$(cut -d' ' -f1 $B/SOURCE)
case "$head" in "$H"*) ;; *) echo "binaries ($head) and tree ($H) differ: stop"; exit 2 ;; esac
gunzip -c $RCP/vendor-cuda-capital.tsv.gz > $M/vendor-cuda-capital.tsv
(cd $B && sha256sum memra-server memra dsv41-run) > $OUT/bins.sha256
echo "head $head, cards: $CUDA_VISIBLE_DEVICES, start $(date +%T)" | tee $OUT/run.txt
env DSV41_RUN=$B/dsv41-run MEMRA_DSV41_TOKEN_MAP=$TM $B/memra model verify checkpoint DeepSeek-V4.1-Flash-1017 \
  --against deepseek_v41 --oracle $M/vendor-cuda-capital.tsv --native-runner $S/scripts/dsv41-run-experts-host \
  --out $OUT > $OUT/verify-checkpoint-res.log 2>&1
echo "checkpoint rc $? $(grep -o 'verified=[A-Za-z]*' $OUT/verify-checkpoint-res.log | tail -1) $(date +%T)" | tee -a $OUT/run.txt
env MEMRA_DSV41_TOKEN_MAP=$TM MEMRA_SERVE_VERIFY_TIMEOUT_S=1500 $B/memra model verify serve DeepSeek-V4.1-Flash-1017 \
  --against deepseek_v41 --out $OUT --native-runner $B/memra-server > $OUT/verify-serve.log 2>&1
echo "serve rc $? $(grep -o 'verified=[A-Za-z]*' $OUT/verify-serve.log | tail -1) $(date +%T)" | tee -a $OUT/run.txt
cat $OUT/gates.txt 2>/dev/null | tee -a $OUT/run.txt
echo "M1-DONE $(date +%T)" | tee -a $OUT/run.txt
