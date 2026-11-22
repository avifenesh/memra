#!/bin/bash
# Declared-class checkpoint parity, the GPU part (run from the repo root of the lane worktree).
# usage: run_gpu.sh route|harness|p3
# route:   per-MoE reference check (forced inputs), float trunk vs the f64 tapes, the probe tape of
#          the runner contract, and the route's head-normalized rows on every corpus document
# harness: lane F's klh.py on the probe and the corpora (parity_kl.py)
set -uo pipefail
L=$HOME/.cache/inkling2mini
B=${INK2_GPU:-$L/target-serve/release/ink2-gpu}
C=$HOME/models/inkling2-mini
T=$L/oracle
CP=research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity
HD=$L/cp-hidden
K=$L/klsens
PROBE=$(sed -n 's/^tokens\t//p' $CP/probe/lane-e-cpu-actq.tsv | tr ',' ' ')
mkdir -p $CP/logs $HD
step() { name=$1; shift; echo "== $name $(date +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader)"; "$@" > $CP/logs/$name.log 2>&1; r=$?; echo "   rc=$r"; tail -n 2 $CP/logs/$name.log; return 0; }
case $1 in
route)
  step p1-moe-ref $B $C --class shipped --tape $T/tapes/p1_chat.actq_f32 --score --force-moe-in --moe-ref 1 --out $CP/p1-moe-ref.json
  step p2-moe-ref $B $C --class shipped --tape $T/tapes/p2_window.actq_f32 --score --force-moe-in --moe-ref 16 --out $CP/p2-moe-ref.json
  step p2-float-vs-f64 $B $C --class float --tape $T/tapes-var/p2.exact_f64.chunk512.as_f32 --score --out $CP/p2-float-vs-f64.json
  step p3-float-vs-f64 $B $C --class float --tape $T/tapes-var/p3.exact_f64.chunk1024.as_f32 --score --chunk 1024 --out $CP/p3-float-vs-f64.json
  step probe-native env MEMRA_ORACLE_OUT=$CP/probe/native-route.tsv MEMRA_FULL_PREC=1 $B $C $PROBE
  step dump-oracle $B $C --class shipped --dump-hidden $CP/oracle-prompts.json $HD/oracle
  step dump-p3 $B $C --class shipped --dump-hidden $CP/oracle-p3.json $HD/p3
  step dump-rawdoc $B $C --class shipped --dump-hidden $K/corpus/rawdoc.json $HD/rawdoc
  step dump-chatwrap $B $C --class shipped --dump-hidden $K/corpus/chatwrap.json $HD/chatwrap
  ;;
harness)
  step harness $K/venv/bin/python $CP/parity_kl.py --probe-oracle "$(echo $PROBE | tr ' ' ',')" $CP/probe \
    --set oracle $CP/oracle-prompts.json $HD/oracle --set rawdoc $K/corpus/rawdoc.json $HD/rawdoc \
    --set chatwrap $K/corpus/chatwrap.json $HD/chatwrap \
    --tape-cross $T/tapes/p1_chat.actq_f32 $HD/oracle 0 --tape-cross $T/tapes/p2_window.actq_f32 $HD/oracle 1 \
    --save-hidden $L/cp-klh-gpu --peer $L/cp-klh-cpu --out $CP/kl-harness.json
  ;;
p3)
  step harness-p3 $K/venv/bin/python $CP/parity_kl.py --set p3 $CP/oracle-p3.json $HD/p3 --out $CP/kl-harness-p3.json
  ;;
esac
echo "RUN_GPU_DONE $1"
