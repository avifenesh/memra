#!/bin/bash
# Lane F's harness on the GPU against the s4 route dumps: nvfp4-rig (probe tapes, the corpora with the
# harness's CPU run as peer) and the shipped class (the corpora and lane E's tapes).
set -uo pipefail
L=$HOME/.cache/inkling2mini
K=$L/klsens
T=$L/oracle/tapes
CP=research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity
HD=$L/cp-hidden-s4
PROBE=$(sed -n 's/^tokens\t//p' $CP/probe/lane-e-cpu-actq.tsv)
mkdir -p $CP/logs $CP/probe-rig
step() { name=$1; shift; echo "== $name $(date +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader)"; "$@" > $CP/logs/$name.log 2>&1; r=$?; echo "   rc=$r"; tail -n 2 $CP/logs/$name.log; return 0; }
step harness-rig $K/venv/bin/python $CP/parity_kl.py --class nvfp4-rig --probe-oracle "$PROBE" $CP/probe-rig \
  --set oracle $CP/oracle-prompts.json $HD/nvfp4-rig/oracle --set rawdoc $K/corpus/rawdoc.json $HD/nvfp4-rig/rawdoc \
  --set chatwrap $K/corpus/chatwrap.json $HD/nvfp4-rig/chatwrap --peer $L/cp-klh-cpu-rig --out $CP/kl-harness-rig.json
step harness-shipped-s4 $K/venv/bin/python $CP/parity_kl.py --class shipped \
  --set oracle $CP/oracle-prompts.json $HD/shipped/oracle --set rawdoc $K/corpus/rawdoc.json $HD/shipped/rawdoc \
  --set chatwrap $K/corpus/chatwrap.json $HD/shipped/chatwrap \
  --tape-cross $T/p1_chat.actq_f32 $HD/shipped/oracle 0 --tape-cross $T/p2_window.actq_f32 $HD/shipped/oracle 1 --out $CP/kl-harness-shipped-s4.json
echo RUN_GPU_S4H_DONE
