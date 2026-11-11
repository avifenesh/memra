#!/usr/bin/env bash
# bs1 CheckpointParity on the pack's capital probe against the vendor CUDA oracle (one lease)
set -u
R=$HOME/.cache/dsv41-bsmma
D=$HOME/.cache/dsv41-speed/data
C=$HOME/models/DeepSeek-V4.1-Flash-1017
TM=$D/oracle/engram_constants.token_map.i32
export MEMRA_FULL_PREC=1
B=$R/bins/bs1
P=$R/out/ckpt-bs1; rm -rf $P/verify-capital; mkdir -p $P
sha256sum $B/dsv41-run $B/memra > $P/bins.sha256
echo "parity start $(date +%T)" > $P/runs.txt
MEMRA_DSV41_TOKEN_MAP=$TM timeout 28m $B/memra model verify checkpoint $C --against deepseek_v41 \
  --oracle $R/vendor-cuda-capital.tsv --native-runner $B/dsv41-run --out $P/verify-capital > $P/verify-capital.log 2>&1
echo "parity rc $? end $(date +%T)" >> $P/runs.txt
grep -E '^(status|max_abs|reference_argmax|native_argmax)' $P/verify-capital/checkpoint-parity.tsv >> $P/runs.txt 2>&1
echo "PARITY-BS1-DONE $(date +%T)" >> $P/runs.txt
