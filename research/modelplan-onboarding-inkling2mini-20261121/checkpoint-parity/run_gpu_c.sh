#!/bin/bash
# Third lease: the reference check's red arm on p1, then lane F's harness on the probe and corpora.
set -uo pipefail
L=$HOME/.cache/inkling2mini
B=${INK2_GPU:-$L/target-serve/release/ink2-gpu}
C=$HOME/models/inkling2-mini
T=$L/oracle
CP=research/modelplan-onboarding-inkling2mini-20261121/checkpoint-parity
step() { name=$1; shift; echo "== $name $(date +%H:%M:%S) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader)"; "$@" > $CP/logs/$name.log 2>&1; r=$?; echo "   rc=$r"; tail -n 2 $CP/logs/$name.log; return 0; }
step p1-moe-ref-red $B $C --class shipped --tape $T/tapes/p1_chat.actq_f32 --score --force-moe-in --moe-ref 1 --red moe-ref-index-order --out $CP/p1-moe-ref-red.json
bash $CP/run_gpu.sh harness
