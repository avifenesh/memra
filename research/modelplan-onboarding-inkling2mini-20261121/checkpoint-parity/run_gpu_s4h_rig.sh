#!/bin/bash
# The nvfp4-rig harness run again, with the harness's CPU run complete as the peer (s4h's first run
# stopped on a peer that covered 28 of 32 documents).
set -uo pipefail
L=$HOME/.cache/inkling2mini
K=$L/klsens
CP=research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity
HD=$L/cp-hidden-s4
step() { name=$1; shift; echo "== $name $(date +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader)"; "$@" > $CP/logs/$name.log 2>&1; r=$?; echo "   rc=$r"; tail -n 2 $CP/logs/$name.log; return 0; }
waitfor --grep RIG_CPU_DONE /home/avifenesh/.cache/inkling2mini/serve-logs/rig-cpu.out --timeout 20m || echo "the CPU peer is not complete; documents without it are scored without a peer"
step harness-rig $K/venv/bin/python $CP/parity_kl.py --class nvfp4-rig \
  --set oracle $CP/oracle-prompts.json $HD/nvfp4-rig/oracle --set rawdoc $K/corpus/rawdoc.json $HD/nvfp4-rig/rawdoc \
  --set chatwrap $K/corpus/chatwrap.json $HD/nvfp4-rig/chatwrap --peer $L/cp-klh-cpu-rig --out $CP/kl-harness-rig.json
echo RUN_GPU_S4H_RIG_DONE
